import os
import pickle
from Bio import SeqIO
import tqdm
import numpy as np

def load_dict(pickle_file):
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

def time_check(info, time_range=None):
    start_time = info['start_time'] 
    if start_time>=time_range[0] and start_time<=time_range[1]:
        return True
    else:
        return False

def target_check(info, target_pred):
    pred = info['Predictions']    
    if pred is None or np.isnan(pred).any():
        return False    
    if int(pred) in target_pred:
        return True
    else:
        return False

def filter_fastq_files(input_dir, output_dir, pickle_file, target_pred, time_range):    
    os.makedirs(output_dir, exist_ok=True)
    
    print('load dict...')
    read_dict = load_dict(pickle_file)
    
    input_file_list = list(filter(lambda filename: filename[-6:] == '.fastq', os.listdir(input_dir)))
    
    with tqdm.tqdm(total=len(input_file_list), mininterval=3.0) as pbar:
        for fastq_file in input_file_list:

            output_fastq = os.path.join(output_dir, os.path.basename(fastq_file))
            
            with open(output_fastq, 'w') as output_handle:
                for record in SeqIO.parse(os.path.join(input_dir,fastq_file), "fastq"):
                    read_id = record.id
                    if read_id in read_dict:
                        info = read_dict[read_id]
                        
                        if target_check(info, target_pred):
                            if time_check(info, time_range):
                                SeqIO.write(record, output_handle, "fastq")
            
            pbar.update(1)

if __name__ == "__main__":
    input_dir = "/data/biolab-nvme-pool1/lijy/guppy_basecalled/sustechseqs_230608/pass/"  
    output_dir = "/data/biolab-nvme-pool1/lijy/sim_readuntil_fastq/CNNLSTMselnet_sustag384_t0.20finetuned_0_5m/"  
    in_pickle_file = "/home/lijy/workspace/sustag/sustechseqs_test/CNNLSTMselnet_sustag384_pred_dict_t0.20finetuned.P" 
    
    target_pred = [i for i in range(29, 43)]  # start from 1 (sustag384)
    # time_range = (0, 1*60*60)  # 0 - 1h
    time_range = (0, 0.5/6*60*60)  # 0 - 5min
    
    filter_fastq_files(input_dir, output_dir, in_pickle_file, target_pred, time_range)
