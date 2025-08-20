import multiprocessing as mp
import h5py
import os
from tqdm import tqdm
import pickle
from sklearn.preprocessing import MinMaxScaler
import scipy
import numpy as np
from typing import List
import random

sample_stpos = 0
num_sampling_sigals = 3000
sample_rate = 4000
mapping_quality_threshold = 8

def listener(processed_file_cnt: mp.Queue, total_file):
    pbar = tqdm(total = total_file) 
    while True:
        if not processed_file_cnt.empty():
            k=processed_file_cnt.get()
            if k==1:
                pbar.update(1)
            else:
                break
    pbar.close()

def barcode_number_label(barcode_number):
    if barcode_number <= 400:
        jiq384 = barcode_number - 1
        jiq96 = -1
    else:
        jiq384 = -1
        jiq96 = barcode_number - 400 - 1

    ont = (barcode_number - 1) % 400 % 96
    porcupine = ((barcode_number-1) %
                 400 + ((barcode_number-1) % 400) // 96) % 96

    return jiq96, jiq384, ont, porcupine

def locate_barcode_endpos(map_st, map_end, move, stride, first_sample):
    start_col = np.arange(first_sample, first_sample + stride * (len(move)+1), stride)

    barcode_len = 34
    primer_len = 23
    ont_len = 24
    porcupine_len = 40

    # locate where the barcode's stpos & endpos are
    base_cnt = 0
    jiq_st = 0
    jiq_end = 0
    barcode_end = 0
    for idx in range(len(move)):
        if base_cnt == map_st+primer_len:
            barcode_st_pos = idx
            jiq_st = idx
        if base_cnt == map_st+primer_len+barcode_len:
            jiq_end = idx
        if base_cnt == map_st+primer_len+barcode_len+8+ont_len+8+porcupine_len:
            barcode_end = idx
        base_cnt += move[idx]
    
    return start_col[barcode_st_pos], start_col[jiq_st], start_col[jiq_end], start_col[barcode_end]


def crop(data, crop_st, crop_end, smooth_len):

    # print(crop_st, crop_end)
    
    transition = np.linspace(data[crop_st], data[crop_end], smooth_len)
    
    smoothed_data = np.concatenate([data[:crop_st], transition, data[crop_end:]])

    return smoothed_data


def process_hdf5(inputpath, out_path, split_file_list, TOTAL: mp.Value, lock_4_total: mp.Lock, 
                 processed_file_cnt: mp.Queue, seq_cnt_table: mp.Array, lock_4_seq_cnt_table: mp.Lock,
                 MAPPED_POSITIVE: mp.Value, lock_4_mapped_positive: mp.Lock, OTHERS: mp.Value, lock_4_others: mp.Lock):
    for i in range(len(split_file_list)):
        curfile = split_file_list[i]
        fin = h5py.File(os.path.join(inputpath, curfile), 'r')
        fout = h5py.File('%s%s.hdf5' % (out_path, curfile[:-6]), 'w')
        for index, read_name in enumerate(fin.keys()):
            with lock_4_total:
                TOTAL.value += 1
            # origin read_name be like 'read_xxxxxx', so remove the first 5 chars
            read_id = read_name[5:]                        
            
            read_info = fin[read_name]

            read_signals = read_info['Raw']['Signal'][:]
            block_stride = read_info['Analyses/Basecall_1D_000/Summary/basecall_1d_template/'].attrs['block_stride']
            move = read_info['Analyses/Basecall_1D_000/BaseCalled_template/Move'][:]
            duration = read_info['Analyses/Segmentation_000/Summary/segmentation/'].attrs['duration_template']
            first_sample = read_info['Analyses/Segmentation_000/Summary/segmentation/'].attrs['first_sample_template']
            num_events = read_info['Analyses/Segmentation_000/Summary/segmentation/'].attrs['num_events_template']
            read_dic = reads_dic.get(read_id, None)

            if read_dic != None and (read_dic['R_s'] == '+' and read_dic['M_q'] >= mapping_quality_threshold):
            # if read_dic != None:
                barcode_number = int(read_dic['T_n'][-3:])
                jiq96, jiq384, ont, porcupine =  barcode_number_label(barcode_number)

                barcode_st, jiq_st, jiq_end,barcode_end = locate_barcode_endpos(read_dic['Q_s'],read_dic['Q_e'],move, block_stride, first_sample)

                if jiq_end > sample_stpos+num_sampling_sigals:
                    continue
                if barcode_st==0 or jiq_st==0 or jiq_end==0 or barcode_end==0:
                    continue
                
                croped_len = 0

                # croped_data = random_crop(read_signals, int(jiq_end), int(barcode_end), 5, num_sampling_sigals)
                croped_data = crop(read_signals, int(jiq_end), int(barcode_end), 5)
                croped_len = int(barcode_end) - int(jiq_end) - 5

                # plot_signal_data(read_signals, croped_data,(jiq_st,barcode_end),(jiq_st, barcode_end - croped_len))

                if croped_data.shape[0] >= sample_stpos+num_sampling_sigals:

                    if jiq96 != -1 and seq_cnt_table[jiq96+1] < balance_max:    # Seq 96
                        with lock_4_seq_cnt_table:
                            seq_cnt_table[jiq96+1] += 1
                        with lock_4_mapped_positive:
                            MAPPED_POSITIVE.value += 1
                        
                        read_group = fout.create_group(read_id)

                        read_group.create_dataset('raw_signal',data=croped_data[sample_stpos:sample_stpos+num_sampling_sigals])
                        info_group = read_group.create_group('other_info')
                        info_group.attrs['barcode_label_class'] = 'jiq96'
                        info_group.attrs['barcode_label'] = jiq96 + 1
                        info_group.attrs['num_sampling_sigals'] = num_sampling_sigals
                        info_group.attrs['sampling_rate'] = sample_rate
                        info_group.attrs['seq_start'] = jiq_st
                        info_group.attrs['seq_end'] = jiq_end
                        info_group.attrs['barcode_st'] = barcode_st
                        info_group.attrs['barcode_end'] = barcode_end
                        info_group.attrs['saved_file'] = curfile

                # break
            
            elif OTHERS.value <= (MAPPED_POSITIVE.value) / 2 and read_signals.shape[0] >= sample_stpos+num_sampling_sigals:
                with lock_4_others:
                    OTHERS.value += 1

                read_group = fout.create_group(read_id)

                read_group.create_dataset('raw_signal',data=read_signals[sample_stpos:sample_stpos+num_sampling_sigals])
                info_group = read_group.create_group('other_info')
                info_group.attrs['barcode_label_class'] = 'jiq96'
                info_group.attrs['barcode_label'] = -1
                info_group.attrs['num_sampling_sigals'] = num_sampling_sigals
                info_group.attrs['sampling_rate'] = sample_rate
                info_group.attrs['seq_start'] = 0
                info_group.attrs['seq_end'] = 0
                info_group.attrs['barcode_st'] = 0
                info_group.attrs['barcode_end'] = 0
                info_group.attrs['saved_file'] = curfile

            
        fin.close()
        fout.close()
        processed_file_cnt.put(1)



if __name__ == "__main__":
    input_guppyfast5_paths = [
        r'/data/biolab-nvme-pool1/lijy/sustag_orctrl/guppy_basecalled/20220427_barcode_34h/workspace/',      # Guppy fast5 dir
	     ]
    
    input_dic_file = r'/data/biolab-nvme-pool1/lijy/sustag_orctrl/barcode_mix_paf_readsdic_all_mapont.P'     # read dictionary file

    df = open(input_dic_file, 'rb')
    reads_dic = pickle.load(df)
    df.close()

    balance_min = 2000
    balance_max = 4000

    out_path = f'/data/biolab-nvme-pool1/lijy/sustag_orctrl/sustag96_1d{num_sampling_sigals}masked_max{balance_max}/'  # output path
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    num_processes = 10
    num_class = 101

    seq_cnt_table = mp.Array("i", num_class)
    lock_4_seq_cnt_table = mp.Lock()
    for _ in range(num_class):
        seq_cnt_table[_] = 0
    TOTAL = mp.Value("i", 0)
    lock_4_total = mp.Lock()
    MAPPED_POSITIVE = mp.Value("i", 0)
    lock_4_mapped_positive = mp.Lock()
    OTHERS = mp.Value("i", 0)
    lock_4_others = mp.Lock()


    for inputpath in input_guppyfast5_paths:
        filelist = os.listdir(inputpath)
        file_raw_list = list(
            filter(lambda filename: filename[-6:] == '.fast5', filelist))
        
        num_files = len(file_raw_list)
        
        processed_file_cnt = mp.Queue()
        # lock_4_filecnt = 

        print(TOTAL.value)
        
        
        avg_size = len(file_raw_list) // num_processes
        remainder = len(file_raw_list) % num_processes

        start = 0
        processes = []
        total_file = len(file_raw_list)
        listen = mp.Process(target=listener, args=(processed_file_cnt, total_file))
        listen.start()
        for i in range(num_processes):
            if i < remainder:
                end = start + avg_size + 1
            else:
                end = start + avg_size
            split_file_list = file_raw_list[start:end]
            process = mp.Process(target=process_hdf5, args=(inputpath, out_path, split_file_list, 
                                                            TOTAL, lock_4_total, processed_file_cnt,
                                                            seq_cnt_table, lock_4_seq_cnt_table,MAPPED_POSITIVE, 
                                                            lock_4_mapped_positive, OTHERS, lock_4_others))
            
            processes.append(process)
            process.start()
            start = end
        
        # 等待所有进程完成
        for process in processes:
            process.join()
        
        processed_file_cnt.put(-1)
        listen.join()
    
    print(MAPPED_POSITIVE.value, OTHERS.value, TOTAL.value)
    for i in range(len(seq_cnt_table)):
        if seq_cnt_table[i] < balance_min:
            print(i, seq_cnt_table[i])