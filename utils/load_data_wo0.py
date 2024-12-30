import torch
import numpy as np
import h5py
import tqdm
import os
import random
import pickle
import re
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

def torch_data_loader_pickle_wo0(input_pickle_file, batch_size, val_rate, total_rate=1.0, is_shuffle=True, norm=False, silence=False):
    df = open(input_pickle_file, 'rb')
    DATA = pickle.load(df)
    df.close()
    X = DATA['x_train']
    Y = DATA['y_train']
    if is_shuffle:
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]
    X = X[:int(len(X)*total_rate)]
    Y = Y[:int(len(Y)*total_rate)]
    
    Y = Y.squeeze()
        
    x_train = np.array(X[int(len(X)*val_rate):],dtype=np.float32)
    y_train = np.array(Y[int(len(Y)*val_rate):],dtype=int)
    x_val = np.array(X[:int(len(X)*val_rate)],dtype=np.float32)
    y_val = np.array(Y[:int(len(Y)*val_rate)],dtype=int)
    
    # reset label range from [1, 96] to [0, 95]
    y_train = y_train - 1
    y_val = y_val - 1
    
    print('x_train.shape:', x_train.shape)
    
    print('x_train max: ',np.max(x_train),'x_train min: ', np.min(x_train))
    
    print('>2000: ', np.sum(x_train > 2000), '<0: ', np.sum(x_train < 0))

    
    
    if not silence:
        print('train shape x:')
        print(x_train.shape)
        print(x_train[0])
        print('train shape y:')
        print(y_train.shape)
        print(y_train[0])

        print('train: ',len(np.unique(y_train)))
        print('validation: ',len(np.unique(y_val)))
    
    if norm:
        scaler = StandardScaler()
        normalized_data = np.zeros_like(x_train)
        for i in range(x_train.shape[0]):
            normalized_data[i] = scaler.fit_transform(x_train[i].reshape(-1, 1)).reshape(1, 3000)

        print('normalized_data: ', normalized_data.shape)
        print(normalized_data[0])

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)

    
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=8)

    return train_loader, val_loader

def torch_data_loader_pickle_test(input_pickle_file, n_classes, batch_size, val_rate, is_shuffle=True):
    df = open(input_pickle_file, 'rb')
    DATA = pickle.load(df)
    df.close()
    X = DATA['x_val']
    Y = DATA['y_val']
    if is_shuffle:
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]
    
    Y = Y.squeeze()
    
    x_test = np.array(X,dtype=np.float32)
    y_test = np.array(Y,dtype=int)
    
    # reset label range from [1, 96] to [0, 95]
    y_test = y_test - 1

    print('test shape x:')
    print(x_test.shape)
    print('test shape y:')
    print(y_test.shape)

    print('test: ',len(np.unique(y_test)))

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    # ont_hot_train = torch.zeros(y_test.shape[0], n_classes).scatter(1,y_test,1)

    # test_dataset = TensorDataset(x_test, ont_hot_train)
    
    test_dataset = TensorDataset(x_test, y_test)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4)

    return test_loader
