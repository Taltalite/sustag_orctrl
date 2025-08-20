# file: my_statistics.py

import logging
import csv
import time
from threading import Lock

class Statistics:

    def __init__(self, output_csv_path: str):
        self.output_path = output_csv_path
        self.start_time = time.time()
        
        self.total_reads_processed = 0
        self.target_reads_retained = 0
        self.nontarget_reads_unblocked = 0
        self.nontarget_reads_retained_override = 0 # e.g., due to dry_run or other overrides
        
        self._lock = Lock()
        
        try:
            self.csv_file = open(self.output_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.header = [
                'timestamp', 'channel', 'read_id', 'seen_count', 
                'decision', 'action', 'action_overridden'
            ]
            self.csv_writer.writerow(self.header)
            logging.info(f"Statistics will be saved to: {self.output_path}")
        except IOError as e:
            logging.error(f"Unable to open or write statistics file {self.output_path}: {e}")
            self.csv_file = None
            self.csv_writer = None

    def log_read(self, **kwargs):
        with self._lock:
            self.total_reads_processed += 1
            
            decision = kwargs.get('decision', 'UNKNOWN')
            action = kwargs.get('action', 'UNKNOWN')

            if decision == "TARGET":
                self.target_reads_retained += 1
            elif decision == "NON_TARGET":
                if action == "UNBLOCK":
                    self.nontarget_reads_unblocked += 1
                else: # e.g., action is RETAIN due to an override
                    self.nontarget_reads_retained_override += 1
            
            if self.csv_writer:
                row = [
                    f"{time.time():.2f}",
                    kwargs.get('channel', -1),
                    kwargs.get('read_id', 'N/A'),
                    kwargs.get('seen_count', -1),
                    decision,
                    action,
                    kwargs.get('action_overridden', False)
                ]
                self.csv_writer.writerow(row)
    
    def flush(self):
        with self._lock:
            if self.csv_file:
                self.csv_file.flush()

    def report_summary(self):
        with self._lock:
            run_duration = time.time() - self.start_time
            reads_per_sec = self.total_reads_processed / run_duration if run_duration > 0 else 0

            # Preventing division by zero
            if self.total_reads_processed == 0:
                unblock_efficiency = 0
            else:
                total_nontargets = self.nontarget_reads_unblocked + self.nontarget_reads_retained_override
                unblock_efficiency = (self.nontarget_reads_unblocked / total_nontargets * 100) if total_nontargets > 0 else 100

            summary = f"""
            ==================================================
                    Experimental statistical summary
            ==================================================
            Total runtime: {run_duration:.2f} sec.
            Tatal Reads: {self.total_reads_processed} (Avg. {reads_per_sec:.2f} reads/sec.)
            --------------------------------------------------
            Target Reads (Retained): {self.target_reads_retained}
            Non-target Reads (Unblocked): {self.nontarget_reads_unblocked}
            Non-target Reads (Retained due to overriding rules): {self.nontarget_reads_retained_override}
            --------------------------------------------------
            Unblock Efficiency: {unblock_efficiency:.2f}%
            ==================================================
            """
            print(summary)
            logging.info(summary)
            
    def close(self):
        if self.csv_file:
            self.csv_file.close()
            logging.info("The statistics file is closed.")