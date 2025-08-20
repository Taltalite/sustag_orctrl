# file: run_adaptive_sampling.py

import argparse
import logging
import time
import toml
from pathlib import Path
from minknow_api.manager import Manager

from readuntil_api import ReadUntilClient, AccumulatingCache
from analysis_worker import Analysis
from statistic import Statistics

def get_connected_devices(host = "127.0.0.1", port = None):
    manager = Manager(host=host, port=port)
    positions = {}
    for position in manager.flow_cell_positions():
        positions[position.name] = position
    return positions

def main():
    parser = argparse.ArgumentParser(description="Customize the entry program for real-time adaptive sampling experiments.")
    parser.add_argument("--toml", required=True, type=Path, help="Path to the experimental TOML configuration file.")
    parser.add_argument("--device", help="Target sequencing device ID (e.g. 'MN12345').")
    parser.add_argument("--host", default="127.0.0.1", help="MinKNOW gRPC server address.")
    parser.add_argument("--port", type=int, default=8000, help="MinKNOW gRPC port.")
    parser.add_argument("--log-file", default="experiment.log", help="The log file path.")
    parser.add_argument("--stats-file", default="experiment_stats.csv", help="The path to the CSV file where the detailed statistics will be saved.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(args.log_file, mode='w',encoding='utf-8'), logging.StreamHandler()]
    )
    logger = logging.getLogger("MainApp")
    logger.info(f"Adaptive sampling start, parameters: {args}")

    try:
        config = toml.load(args.toml)
        logger.info(f"Configuration file loaded successfully: {args.toml}")
    except Exception as e:
        logger.error(f"Unable to load TOML configuration file: {e}")
        return 1

    stats_manager = Statistics(output_csv_path=args.stats_file)
    
    
    seq_devices = get_connected_devices()
    logging.info(seq_devices)
    flow_cell_position = seq_devices[args.device]
    logging.info('flowcell_connect_port: ' + str(flow_cell_position.rpc_ports.secure))
    
    try:
        client = ReadUntilClient(
            mk_host=flow_cell_position.host,
            mk_port=flow_cell_position.rpc_ports.secure,
            cache_type=AccumulatingCache,
            max_raw_signal = 15000,    # max signal points
            first_channel=1,
            last_channel=256,
            filter_strands=False,
            one_chunk=False,
            calibrated_signal=False
        )
        logger.info(f"Successfully connected to the MinKNOW gRPC server at {args.host}:{args.port}")
        
        logger.debug(f"client.last_channel: {client.last_channel}")

        analysis_session = Analysis(client, config, logging.getLogger("AnalysisSession"), statistics=stats_manager)

        analysis_session.run(batch_size=50)

    except KeyboardInterrupt:
        logger.info("Manual interrupt (Ctrl+C) detected.")
    except Exception as e:
        logger.error(f"Fatal error occurred, and the experiment was terminated: {e}", exc_info=True)
    finally:
        if 'session' in locals() and analysis_session.is_running:
            analysis_session.stop()
        
        stats_manager.report_summary()
        stats_manager.close()
        logger.info("The experiment is completely over.")

if __name__ == "__main__":
    main()