import torch
from torch import nn
import sys
sys.path.append('/home/lijy/workspace/')
from sustag_orctrl.model.model_CNNLSTM import CNN_LSTM
from sustag_orctrl.utils.load_data import torch_data_loader_pickle
from torchinfo import summary
import numpy as np
import os

import time
import datetime
import random


# input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/ONT_1d3000masked_max4000/train_set/barcode_all_1d3000_ONT_100_1wun.P'
input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/seq96_1d3000masked_max4000/train_set/barcode_all_1d3000_sustag96_100_4kun.P'
# input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/Porcupine_1d3000masked_max4000/train_set/barcode_all_1d3000_Porcupine_100_1wun.P'
  

def model_summary(model):
    summary(model, depth=3, input_size=(batch_size, 1, 3000))
    return model
    


def train(num_class: int, hidden_dim: int, batch_size: int, epochs: int, learning_rate: float,
          weight_decay: float, dropout_rate:float, val_rate:float,SAVE_MODEL:bool, current_device):
    
    model = CNN_LSTM(num_class, dropout_rate, hidden_dim).to(current_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss = nn.CrossEntropyLoss()
    
    model_summary(model)

    total_rate = 1.0
    print('input_pickle_file: ', input_pickle_file)
    train_loader, val_loader = torch_data_loader_pickle(
            input_pickle_file, batch_size, val_rate,total_rate=total_rate, is_shuffle=True, silence=False)
    size = len(train_loader.dataset)
    print('train_loader size: ', size)    
    
    ckpt_path = '../checkpoints/CNNLSTM/'
    os.makedirs(ckpt_path, exist_ok=True)
    model_out_path = f"{ckpt_path}/CNNLSTM_sustag96_e{epochs}_b{batch_size}_t{total_rate:>.2f}_4kun.pth"
    print('model save path: ', model_out_path)

    max_acc = 0
    min_loss = np.inf

    for epoch in range(1, epochs+1):
        start_time = time.time()  # 记录epoch开始的时间
        # set model to train mode
        model.train()

        for batch_i, (X, Y) in enumerate(train_loader):
            batches_done = len(train_loader) * epoch + batch_i

            X = X.to(current_device)
            Y = Y.to(current_device)
            Y_hat = model(X)
            
            l = loss(Y_hat, Y)
            l.backward()
            # Run optimizer
            optimizer.step()
            # Reset gradients
            optimizer.zero_grad()

            if batch_i % int((size//batch_size)*0.4) == 0:
                l, current = l.item(), batch_i * len(X)
                print(f"loss: {l:>7f}  [{current:>5d}/{size:>5d}]")

        correct = 0
        total = 0
        # set model to train mode
        model.eval()
        with torch.no_grad():
            total_loss = 0.0  # 用于累计该epoch的总损失
            total = 0  # 总样本数，用于计算平均损失
            for batch_i, (X, Y) in enumerate(val_loader):
                X = X.to(current_device)
                Y = Y.to(current_device)
                Y_hat = model(X)
                val_loss = loss(Y_hat, Y)
                total_loss += val_loss.item() * Y.size(0)  # 累加批次损失，乘以批次中的样本数
                total += Y.size(0)

                pred = torch.argmax(Y_hat, dim=1)
                # labels = torch.argmax(Y, dim=1)
                labels = Y
                correct += (pred == labels).sum().item()

            avg_loss = total_loss / total  # 计算该epoch的平均损失
            cross_acc = 100 * correct / total
            
            # print('avg_loss: %f' % avg_loss)
            # print('cross_acc: %f' % cross_acc)

            if avg_loss < min_loss and SAVE_MODEL:
                min_loss = avg_loss
                print('='*20,'save the best model based on lowest loss','='*20)
                torch.save(model.state_dict(), model_out_path)

                max_acc = cross_acc
                # print('min_loss: %f' % min_loss)
                # print('max_acc: %f' % max_acc)
            
            
        scheduler.step()
        print('last lr:', scheduler.get_last_lr())
        print(
            f'\nAccuracy of the model on the {total} val sets: {cross_acc:>.3f}%    loss: {avg_loss:>.6f}')
        end_time = time.time()  # 记录epoch结束的时间
        epoch_duration = end_time - start_time  # 计算epoch的持续时间
        print(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds")

    return max_acc


if __name__ == "__main__":
    num_gpu_devices = torch.cuda.device_count()
    print("Avaliable GPU devices:", num_gpu_devices)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    hidden_dim = 128
    total_rate = 1.0
    val_rate = 0.05

    num_class = 97
    batch_size = 1024
    epochs=100
    
    # Default
    learning_rate=3* 1e-3
    weight_decay=0.001
    dropout_rate = 0.30

    print('=' * 95)
    print('hidden_dim:%d' % hidden_dim)
    print('num_class:%d' % num_class)
    print('batch_size:%d' % batch_size)
    print('epochs:%d' % epochs)
    print('learning_rate:%f' % learning_rate)
    print('weight_decay:%f' % weight_decay)
    print('dropout_rate:%f' % dropout_rate)
    print('Dataset: %s' % input_pickle_file)
    print('=' * 95)

    
    current_datetime = datetime.datetime.now()
    print("Training start from: ", current_datetime)
    
    SAVE_MODEL = True
    train(num_class,hidden_dim,batch_size,epochs,learning_rate,
          weight_decay,dropout_rate,val_rate,SAVE_MODEL, device)

    current_datetime = datetime.datetime.now()
    print("Training end at: ", current_datetime)
