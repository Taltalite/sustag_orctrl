import os
import logging
import time
import traceback
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from sklearn.preprocessing import StandardScaler
import numpy as np
import toml
import torch
from statistic import Statistics
import sys
sys.path.append('/home/lijy/workspace/')
from sustag_orctrl.model.model_ORCtrL import CNN_LSTM_Backbone, OPRNet, SelectiveLoss

from readuntil_api import ReadUntilClient


class ChunkTracker:
    def __init__(self):
        self.seen_counts = {}
    def seen(self, channel, read_id):
        key = (channel, read_id)
        self.seen_counts.setdefault(key, 0)
        self.seen_counts[key] += 1
        return self.seen_counts[key]

class Analysis:
    def __init__(self, client: ReadUntilClient, config: dict, logger: logging.Logger, statistics: Statistics):
        self.client = client
        self.config = config
        self.logger = logger
        self.is_running = False
        
        self.logger.debug("Initializing Analysis session with the following parameters:")
        self.logger.debug(f"  [general] queue_size = {config['general']['queue_size']}")
        self.logger.debug(f"  [general] analysis_workers = {config['general']['analysis_workers']}")
        self.logger.debug(f"  [acquisition] max_missed_start_offset = {config['acquisition']['max_missed_start_offset']}")
        self.logger.debug(f"  [actions] non_unblock = {config['actions']['non_unblock']}")
        self.logger.debug(f"  [actions] unblock_duration = {config['actions']['unblock_duration']}")
        self.logger.debug(f"  [analysis] is_data_norm = {config['analysis']['is_data_norm']}")
        self.logger.debug(f"  [analysis] min_signal_length = {config['analysis']['min_signal_length']}")
        self.logger.debug(f"  [analysis] sample_start_pos = {config['analysis']['sample_start_pos']}")
        self.logger.debug(f"  [analysis] confidence_threshold = {config['analysis']['confidence_threshold']}")

        # --- Pipeline Queue ---
        queue_size = self.config['general']['queue_size']
        self.processing_queue = Queue(maxsize=queue_size)
        self.decision_queue = Queue(maxsize=queue_size)

        # --- Worker thread pool ---
        num_analysis_workers = self.config['general']['analysis_workers']
        self.analysis_executor = ThreadPoolExecutor(max_workers=num_analysis_workers, thread_name_prefix="AnalysisWorker")
        self.decision_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DecisionWorker")
        self.analysis_queue_timeout = self.config['analysis']['analysis_queue_timeout']
        self.decision_queue_timeout = self.config['analysis']['decision_queue_timeout']


        self.statistics = statistics
        self.chunk_tracker = ChunkTracker()

        # --- Model loading and configuration ---
        self.data_norm = self.config['analysis']['is_data_norm']
        self.min_signal_length = self.config['analysis']['min_signal_length']
        self.sample_start_pos = self.config['analysis']['sample_start_pos']
        self.threshold = self.config['analysis']['confidence_threshold']
        
        self.target_labels = set(self.config['analysis']['target_class_labels'])
        self.logger.info(f"Target category list loaded: {self.target_labels}")
        
        self.numclasses = 385
        self.dropout_rate = 0.30
        self.hidden_dim = 128
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"The model will run on: {self.device}")
        
        features = CNN_LSTM_Backbone(self.numclasses, self.dropout_rate, self.hidden_dim).to(self.device)
        self.model_orctrl = OPRNet(features, 37 * self.hidden_dim, self.numclasses, dropout_rate=self.dropout_rate).to(self.device)    
        self.model_orctrl.load_state_dict(torch.load(os.path.join(self.config['analysis']['model_path']), map_location=self.device))
        self.model_orctrl.eval()
        
        self.logger.info("Analysis initialized.")

    def _analysis_worker(self):
        """
        Get data from the queue, preprocess the signal, and use the DL model for classification.
        """
        self.logger.info("The analysis worker is started.")
        scaler = StandardScaler()
        while self.is_running:
            try:
                channel, read, missed_obs = self.processing_queue.get(timeout=self.analysis_queue_timeout)

                read_id = read.id

                
                raw_signal = np.frombuffer(read.raw_data[-missed_obs:], self.client.signal_dtype)
                
                required_length = self.sample_start_pos + self.min_signal_length
                if raw_signal.shape[0] < required_length:
                    continue
                  
                # Trim the signal to the length required by the model
                X_np = raw_signal[self.sample_start_pos : self.sample_start_pos + self.min_signal_length]
                   
                if self.data_norm:
                    X_reshaped = X_np.reshape(-1, 1)
                    X_normalized_np = scaler.fit_transform(X_reshaped)
                    X_normalized_np = X_normalized_np.flatten()
                    X_tensor = torch.from_numpy(X_normalized_np.copy()).float().unsqueeze(0).unsqueeze(0).to(self.device)
                else:
                    X_tensor = torch.from_numpy(X_np.copy()).float().unsqueeze(0).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    out_class, out_select, out_aux = self.model_orctrl(X_tensor)
                    out_select = out_select.squeeze()
                    class_preds = torch.argmax(out_class, dim=1)

                    prediction = torch.where(out_select >= self.threshold, class_preds, torch.tensor(-1, device=self.device)).cpu().item()
                    # prediction = torch.where(out_select >= self.threshold, class_preds, torch.zeros_like(class_preds)).cpu().item()

                self.logger.debug(f"CH:{channel} ID:{read_id} - Prediction: {prediction}, Confidence: {out_select.item():.2f}")
                
                result = {
                    "channel": channel, 
                    "read_id": read.id,
                    "prediction": prediction,
                    "analysis_summary": f"pred={prediction}, conf={out_select.item():.2f}"
                }
                self.decision_queue.put(result)

            except Empty:
                continue # The queue is empty, continue waiting
            except Exception:
                self.logger.error(f"Analysis worker error: {traceback.format_exc()}")
    
    def _decision_worker(self):
        """
        Makes a final decision and uses read_id to send commands.
        """
        self.logger.info("The decision worker is started.")
        non_unblock = self.config['actions']['non_unblock']
        unblock_duration = self.config['actions']['unblock_duration']
        
        while self.is_running:
            try:
                result = self.decision_queue.get(timeout=self.decision_queue_timeout)
                
                channel, read_id = result['channel'], result['read_id']
                seen_count = self.chunk_tracker.seen(channel, read_id)

                is_target = (result['prediction'] in self.target_labels)
                action = "retain" if is_target else "unblock"
                action_overridden = False
                if action == "unblock" and non_unblock:
                    action = "retain"
                    action_overridden = True
                
                # --- MODIFICATION: Call client methods with read_id ---
                if action == "retain":
                    self.logger.info(f"✅ CH:{channel} ID:{read_id} - Target - {result['analysis_summary']}")
                    self.client.stop_receiving_read(channel, read_id)
                else: # action == "unblock"
                    self.logger.info(f"❌ CH:{channel} ID:{read_id} - Non-target - {result['analysis_summary']}")
                    self.client.unblock_read(channel, read_id, duration=unblock_duration)
                
                # --- 记录统计信息 ---
                self.statistics.log_read(
                    channel=channel, read_id=read_id, seen_count=seen_count,
                    decision="TARGET" if is_target else "NON_TARGET",
                    action=action.upper(), action_overridden=action_overridden
                )

            except Empty:
                continue
            except Exception:
                self.logger.error(f"Decision worker error: {traceback.format_exc()}")


    def start(self):
        if self.is_running:
            self.logger.warning("A session is already running.")
            return

        self.logger.info("Starting session")
        self.is_running = True
        
        num_analysis = self.config['general']['analysis_workers']
        for _ in range(num_analysis):
            self.analysis_executor.submit(self._analysis_worker)
        self.decision_executor.submit(self._decision_worker)

        self.logger.debug(f"self.client.first_channel: {self.client.first_channel}, self.client.last_channel: {self.client.last_channel}")
        self.client.run(
            first_channel=self.client.first_channel,
            last_channel=self.client.last_channel
        )
        
        self.logger.info("The session started successfully.")

    def stop(self):
        if not self.is_running:
            return
            
        self.logger.info("Stopping session...")
        self.is_running = False
        
        self.client.reset()
        
        self.analysis_executor.shutdown(wait=True, cancel_futures=True)
        self.decision_executor.shutdown(wait=True, cancel_futures=True)
        
        self.logger.info("The session stopped successfully.")

    def run(self, batch_size=50):
        try:
            self.start()
            
            min_points_required = self.min_signal_length
            bytes_per_point = np.dtype(self.client.signal_dtype).itemsize
            min_bytes_required = min_points_required * bytes_per_point
            max_missed_offset = self.config['acquisition']['max_missed_start_offset']
            
            last_data_receipt_time = time.time()
            
            while self.is_running:
                if not self.client.is_running:
                    self.logger.warning("ReadUntil client has stopped running and is shutting down...")
                    break
                
                reads = self.client.get_read_chunks(batch_size=batch_size, last=False)
                if reads:
                    current_time = time.time()
                    time_since_last_batch = current_time - last_data_receipt_time
                    self.logger.info(f"Data reporting interval: {time_since_last_batch:.4f} sec. (This batch contains {len(reads)} reads)")
                    last_data_receipt_time = current_time
                    
                    for channel, read in reads:
                        raw_signal = np.frombuffer(read.raw_data, self.client.signal_dtype)
                        missed_obs = read.chunk_start_sample - read.start_sample

                        if missed_obs > max_missed_offset:
                            self.logger.warning(
                                f"CH:{channel} Read:{read.number} - Too many start signal drops ({missed_obs} points), Discarded."
                            )
                            self.client.stop_receiving_read(channel, read.number)
                            continue


                        if missed_obs < 0:
                            raw_signal = raw_signal[-missed_obs:]

                        if raw_signal.shape[0] >= min_points_required:
                            if not self.processing_queue.full():
                                self.processing_queue.put((channel, read, missed_obs))
                            else:
                                self.logger.warning(
                                    f"The analysis queue is full ({self.processing_queue.qsize()}). Drop CH:{channel} read。"
                                )
                else:
                    time.sleep(self.config['general']['throttle'])
        
        finally:
            self.stop()