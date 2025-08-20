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
                 lock_4_listupdate: mp.Lock, filted_list: List):
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
                        if filted_list == []:
                            shared_Y.append([label])
                        else:
                            if label not in filted_list:
                                shared_Y.append([filtering(label, filted_list)])
                            else:
                                continue

                        shared_X.append([source_sig])
                        
            processed_file_cnt.put(1)
        # break



if __name__ == '__main__':
   
    total_rate = 1.0
    test_rate = 0.1
    filted_list = [55, 31, 39, 32] # start from 1 sustag96


    input_rawdata_path = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/seq96_1d3000masked_max4000/'

    
    print('dataset: ', input_rawdata_path)

    out_path = input_rawdata_path+'/train_set/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    

    file_raw_list = [os.path.join(input_rawdata_path, filename) for filename in os.listdir(input_rawdata_path) if filename.endswith('.hdf5')]

    random.shuffle(file_raw_list)
    file_raw_list = file_raw_list[:int(len(file_raw_list)*total_rate)]

    manager = mp.Manager()
    shared_X = manager.list()
    shared_Y = manager.list()

    lock_4_listupdate = mp.Lock()

    processed_file_cnt = mp.Queue()


    # Set proc num
    # num_processes = mp.cpu_count()  
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
                                                        processed_file_cnt,
                                                        lock_4_listupdate, filted_list))

        processes.append(process)
        process.start()
        start = end

    for process in processes:
        process.join()
    
    processed_file_cnt.put(-1)
    listen.join()

    X = np.array(list(shared_X), dtype=np.float16)
    Y = np.array(list(shared_Y), dtype=int)



    if True:
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]

    print(len(np.unique(Y)))
    
    x_train = np.array(X[int(len(X)*test_rate):],dtype=np.float32)
    y_train = np.array(Y[int(len(Y)*test_rate):],dtype=int)
    
    x_test = np.array(X[:int(len(X)*test_rate)],dtype=np.float32)
    y_test = np.array(Y[:int(len(Y)*test_rate)],dtype=int)


    print('train: ',len(np.unique(y_train)))
    print('test: ',len(np.unique(y_test)))


    print('x_train shape: ', x_train.shape)
    print('y_train shape: ', y_train.shape)

    print('x_test shape: ', x_test.shape)
    print('y_test shape: ', y_test.shape)


    DATA = {'x_train':x_train,'y_train':y_train, 'x_val':x_test,'y_val':y_test}

    print('dump to pickle file...')
    
    df = open(os.path.join(out_path,'barcode_all_1d3000_sustag96_%d_wo0.P' % (int(total_rate*100))), 'wb')
    pickle.dump(DATA, df)
    df.close()

    print('done!')



