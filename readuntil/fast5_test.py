# file: fast5_minion_tester.py

import argparse
import logging
import time
import random
import uuid
from queue import Queue
from pathlib import Path

import numpy as np
import toml
import h5py 
from statistic import Statistics

from analysis_worker import Analysis 

class FakeReadData:
    def __init__(self, raw_data: bytes, channel: int, read_number: int, start_sample: int, read_id: str):
        self.raw_data = raw_data
        self.channel = channel
        self.number = read_number
        self.id = read_id
        self.start_sample = start_sample
        self.chunk_start_sample = start_sample 
        self.chunk_length = len(raw_data) // np.dtype(np.int16).itemsize
        self.chunk_classifications = ['strand'] 

class Fast5ReadUntilClient:
    """
    A simulation client that loads real signal data from FAST5 files and simulates real-time streaming.
    """
    def __init__(self, config: dict,
                 first_channel: int = None,
                    last_channel: int = None,
                    num_reads_to_load: int = 500):
        self.config = config
        self.total_reads_to_load = num_reads_to_load
        
        self.first_channel=1
        self.last_channel=512

        self.pending_reads_pool = []

        self.active_channels = {}
        self.generated_reads_count = 0
        
        self.is_running = False
        self.signal_dtype = np.int16

        logging.info(f"Loading FAST5 files from directory {self.config['simulation']['fast5_directory']} ...")
        self._load_data_from_fast5()
        if not self.pending_reads_pool:
            raise RuntimeError("Failed to load any valid FAST5 signal data from the specified directory.")
        logging.info(f"Successfully loaded {len(self.pending_reads_pool)} reads into the simulation pool.")

    def _load_data_from_fast5(self):
        """
        Scans directories, correctly reads multi-read FAST5 files and extracts all raw signals.
        """
        fast5_dir = Path(self.config['simulation']['fast5_directory'])
        if not fast5_dir.is_dir():
            logging.error(f"Error: Directory does not exist -> {fast5_dir}")
            return
            
        fast5_files = list(fast5_dir.glob("*.fast5"))
        random.shuffle(fast5_files)
        
        for f5_path in fast5_files:
            if len(self.pending_reads_pool) >= self.total_reads_to_load:
                break
            try:
                with h5py.File(f5_path, 'r') as f:
                    for read_name in f.keys():
                        if len(self.pending_reads_pool) >= self.total_reads_to_load:
                            break
                        try:
                            signal_path = f'{read_name}/Raw/Signal'
                            signal = f[signal_path][()]
                            
                            if signal.dtype != np.int16:
                                signal = signal.astype(np.int16)

                            self.pending_reads_pool.append(signal)

                        except KeyError:
                            logging.warning(f"Signal path not found in {f5_path.name} in {read_name}, {signal_path}")
                            continue
            except Exception as e:
                logging.warning(f"Errors whrn reading {f5_path.name}: {e}")
    

    def run(self, **kwargs):
        logging.info("Fast5Client: 'run' called. Setting is_running to True.")
        self.is_running = True

    def reset(self):
        logging.info("Fast5Client: 'reset' called. Setting is_running to False.")
        self.is_running = False
        
    def get_read_chunks(self, batch_size=50, last=False):
        if not self.is_running: return []
        reads_batch = []
        
        num_new_reads = random.randint(1, 5)
        for _ in range(num_new_reads):
            if not self.pending_reads_pool: break
            channel = random.randint(1, 512)
            if channel not in self.active_channels:
                full_signal = self.pending_reads_pool.pop(0)
                read_number = self.generated_reads_count + 1
                self.active_channels[channel] = {
                    "full_signal": full_signal, "current_pos": 0,
                    "read_number": read_number, "read_id": str(uuid.uuid4()),
                    "start_sample": int(time.time() * 4000)
                }
                self.generated_reads_count += 1
        
        finished_channels = []
        for channel, read_state in self.active_channels.items():
            start_pos = read_state["current_pos"]
            chunk_size = random.randint(1000, 2000)
            end_pos = start_pos + chunk_size
            accumulated_signal = read_state["full_signal"][:end_pos]
            
            if len(accumulated_signal) > 0:
                fake_read = FakeReadData(
                    raw_data=accumulated_signal.tobytes(), channel=channel,
                    read_number=read_state["read_number"], read_id=read_state["read_id"],
                    start_sample=read_state["start_sample"]
                )
                reads_batch.append((channel, fake_read))
            
            read_state["current_pos"] = end_pos
            if end_pos >= len(read_state["full_signal"]):
                finished_channels.append(channel)

        for channel in finished_channels:
            del self.active_channels[channel]
            
        if not self.pending_reads_pool and not self.active_channels:
             self.is_running = False
             logging.info("All reads loaded from FAST5 have been processed and the simulation is about to end.")
        
        logging.debug(f"Fast5Client: Yielding {len(reads_batch)} chunks from {len(self.active_channels)} active channels.")
        time.sleep(self.config['general']['throttle'])
        return reads_batch

    def unblock_read(self, channel: int, read_number: int, duration: float):
        logging.info(f"Fast5Client: Received UNBLOCK for CH:{channel}, Read:{read_number}. Simulating removal.")
        if channel in self.active_channels:
            del self.active_channels[channel]

    def stop_receiving_read(self, channel: int, read_number: int):
        logging.info(f"Fast5Client: Received STOP_RECEIVING for CH:{channel}, Read:{read_number}")

def main():
    parser = argparse.ArgumentParser(description="FAST5 file-based simulation tester for Read Until processes.")
    parser.add_argument("--toml", required=True, type=Path, help="The path to your experiment TOML configuration file.")
    parser.add_argument("--log-file", default="fast5_test.log", help="Log file path.")
    parser.add_argument("--num-reads", type=int, default=500, help="The maximum number of FAST5 files to load from the directory.")
    parser.add_argument("--stats-file", default="fast5_test_stats.csv", help="The path to the CSV file where the detailed statistics will be saved.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(args.log_file, mode='w',encoding='utf-8'), logging.StreamHandler()])

    logging.info("Starting...")

    config = toml.load(args.toml)
    # config.setdefault('actions', {})['dry_run'] = True
    main_logger = logging.getLogger(__name__)
    stats_manager = Statistics(output_csv_path=args.stats_file)
    try:
        fast5_client = Fast5ReadUntilClient(config, num_reads_to_load=args.num_reads)
                
        analysis_session = Analysis(fast5_client, config, main_logger, statistics=stats_manager)
        
        analysis_session.run(batch_size=50)

    except KeyboardInterrupt:
        logging.info("Manual interrupt (Ctrl+C) detected.")
    except Exception as e:
        logging.error(f"An error occurred while the simulation session was running: {e}", exc_info=True)
    finally:
        if 'analysis_session' in locals() and analysis_session.is_running:
            analysis_session.stop()
        stats_manager.report_summary()
        stats_manager.close()
    
    logging.info("The simulation test is completed.")

if __name__ == "__main__":
    main()