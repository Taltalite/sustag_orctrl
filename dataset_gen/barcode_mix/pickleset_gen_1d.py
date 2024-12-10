import os
from tqdm import tqdm
import h5py
import numpy as np
import pickle
import multiprocessing as mp
from filelock import FileLock
from typing import List
import random


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
    
def filtering(old_label, filter_list):
    cnt = 0
    for i in range(len(filter_list)):
        if old_label > filter_list[i]:
            cnt += 1
    return old_label - cnt


def process_hdf5(file_list: List, shared_X: mp.Manager().list, shared_Y: mp.Manager().list, processed_file_cnt: mp.Queue, 
                 shared_unclassified_cnt: mp.Value, lock_4_unc: mp.Lock, lock_4_listupdate: mp.Lock, max_unclassified: int, 
                 is_binary: bool, shared_Y_bi: mp.Manager().list, filted_list: List):
    if True:
        random.shuffle(file_list)
    for curfile in file_list:
        with h5py.File(curfile, 'r') as h5file:
            for index, read_id in enumerate(h5file.keys()):
                source_sig = h5file['%s/raw_signal'%read_id][:]

                label = h5file['%s/other_info'%read_id].attrs['barcode_label']

                try:
                    label = h5file['%s/other_info'%read_id].attrs['barcode_label']
                except KeyError:
                    print('Rare KeyError!')
                    print('file name: %s, read id: %s' % (curfile, read_id))
                    continue
                if label == 0:
                    print('here is a label being 0, please check')
                    exit(-1)
                if label != -1:
                    with lock_4_listupdate:                        
                        if is_binary:
                            shared_Y.append([1])
                        elif filted_list == []:
                            shared_Y.append([label])
                            shared_Y_bi.append([1])
                        else:
                            if label not in filted_list:
                                shared_Y.append([filtering(label, filted_list)])
                            else:
                                continue
                            shared_Y_bi.append([1])
                        shared_X.append([source_sig])
                elif shared_unclassified_cnt.value < max_unclassified:
                    with lock_4_listupdate:
                        shared_X.append([source_sig])
                        shared_Y.append([0])
                        shared_Y_bi.append([0])
                    with lock_4_unc:
                        shared_unclassified_cnt.value += 1
                else:
                    continue
            processed_file_cnt.put(1)
        # break



if __name__ == '__main__':

    # max_unclassified = 50000
    # max_unclassified = 10000 # about 1% unclassified for 1M data
    max_unclassified = 0
    
    total_rate = 1.0
    test_rate = 0.1
    filted_list = [55, 31, 39, 32] # start from 1 sustag96
    # filted_list = [9,18,19,35,55,69,99,130,151,180,247,291,306,343,375,394] # start from 1 sustag384
    # filted_list = []

    input_rawspec_path = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/seq96_1d3000masked_max4000/'
    # input_rawspec_path = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/seq384_1d1500masked/'
    # input_rawspec_path = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/ONT_1d3000masked_max4000/'
    # input_rawspec_path = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/Porcupine_1d3000masked_max4000/'
    # input_rawspec_path = r'/data/biolab-nvme-pool1/lijy/DNA_storage_data/1d3000/barcode_all_1d3000_100_full/'
    # input_rawspec_path = r'/data/biolab-nvme-pool1/lijy/DNA_storage_data/1d3000/sustseqd_all_1d1500_100_max4000/'
    
    is_binary = False
    print('dataset: ', input_rawspec_path)

    out_path = input_rawspec_path+'/train_set/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    

    file_raw_list = [os.path.join(input_rawspec_path, filename) for filename in os.listdir(input_rawspec_path) if filename.endswith('.hdf5')]

    random.shuffle(file_raw_list)
    file_raw_list = file_raw_list[:int(len(file_raw_list)*total_rate)]

    manager = mp.Manager()
    shared_X = manager.list()
    shared_Y = manager.list()
    shared_Y_bi = manager.list()
    lock_4_listupdate = mp.Lock()

    processed_file_cnt = mp.Queue()

    shared_unclassified_cnt = mp.Value("i", 0)
    lock_4_unc = mp.Lock()

    # 创建一个多进程池
    # num_processes = mp.cpu_count()  # 使用CPU核心数作为进程数
    num_processes = 12

    total_file = len(file_raw_list)
    listen = mp.Process(target=listener, args=(processed_file_cnt, total_file))
    listen.start()

    avg_size = len(file_raw_list) // num_processes
    remainder = len(file_raw_list) % num_processes
    start = 0
    processes = []
    print(num_processes, avg_size, remainder)
    for i in range(num_processes):
        if i < remainder:
            end = start + avg_size + 1
        else:
            end = start + avg_size
        split_file_list = file_raw_list[start:end]
        process = mp.Process(target=process_hdf5, args=(split_file_list, shared_X, shared_Y, 
                                                        processed_file_cnt, shared_unclassified_cnt, lock_4_unc,
                                                        lock_4_listupdate, max_unclassified, is_binary, shared_Y_bi,
                                                        filted_list))

        processes.append(process)
        process.start()
        start = end

    for process in processes:
        process.join()
    
    processed_file_cnt.put(-1)
    listen.join()

    X = np.array(list(shared_X), dtype=np.float16)
    Y = np.array(list(shared_Y), dtype=int)
    Y_bi = np.array(list(shared_Y_bi), dtype=int)



    if True:
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]
        Y_bi = Y_bi[randomize]
    # n_max = int(len(X)*total_rate)
    # X = X[:n_max]
    # Y = Y[:n_max]
        
    # print(np.unique(Y))
    print(len(np.unique(Y)))
    print(len(np.unique(Y_bi)))
    
    x_train = np.array(X[int(len(X)*test_rate):],dtype=np.float32)
    y_train = np.array(Y[int(len(Y)*test_rate):],dtype=int)
    y_train_bi = np.array(Y_bi[int(len(Y_bi)*test_rate):],dtype=int)
    x_test = np.array(X[:int(len(X)*test_rate)],dtype=np.float32)
    y_test = np.array(Y[:int(len(Y)*test_rate)],dtype=int)
    y_test_bi = np.array(Y_bi[:int(len(Y_bi)*test_rate)],dtype=int)

    print('train: ',len(np.unique(y_train)))
    print('test: ',len(np.unique(y_test)))
        
    #     x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
    #     x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))

    print('x_train shape: ', x_train.shape)
    print('y_train shape: ', y_train.shape)
    print('y_train_bi shape: ', y_train_bi.shape)
    print('x_test shape: ', x_test.shape)
    print('y_test shape: ', y_test.shape)
    print('y_test_bi shape: ', y_test_bi.shape)

    DATA = {'x_train':x_train,'y_train':y_train, 'y_train_bi': y_train_bi, 'x_val':x_test,'y_val':y_test, 'y_val_bi': y_test_bi}

    print('dump to pickle file...')
    
    df = open(os.path.join(out_path,'barcode_all_1d3000_sustag384_%d_1wun.P' % (int(total_rate*100))), 'wb')
    pickle.dump(DATA, df)
    df.close()

    print('done!')



