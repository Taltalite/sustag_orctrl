import torch
from torch import nn
import sys
sys.path.append('/home/lijy/workspace/')
# from sustag.model.model_CNN import CNN_1d
from sustag_orctrl.model.model_CNN import CNN_1d
from sustag_orctrl.utils.load_data import torch_data_loader_pickle
from sustag_orctrl.utils.load_data_wo0 import torch_data_loader_pickle_wo0
from torchinfo import summary
import numpy as np
import time
import os
import tqdm
import datetime


# input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/seq96_1d3000masked_max4000/train_set/barcode_all_1d3000_sustag96_100_1wun.P'
# input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/Porcupine_1d3000masked_max4000/train_set/barcode_all_1d3000_Porcupine_100_1wun.P'
# input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/ONT_1d3000masked_max4000/train_set/barcode_all_1d3000_ONT_100_1wun.P'
# input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/seq384_1d1500masked/train_set/barcode_all_1d1500_seq384_100_5wun.P'
# input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/seq384_1d1500masked_withoutun/train_set/barcode_all_1d1500_sustag384_100_0un.P'
input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/seq96_1d3000masked_max4000/train_set/barcode_all_1d3000_sustag96_100_wo0.P'



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)


def train(num_class: int, batch_size: int, epochs: int, learning_rate: float, weight_decay: float, dropout_rate:float,total_rate:float, val_rate:float, device):

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    model_out_path = f"/home/lijy/workspace/sustag_orctrl/checkpoints/CNN/CNN_sustag96_1d3000_masked_ckpt_e{epochs}_b{batch_size}_0un.pth"
    print('model_out_path: ',model_out_path)

    # Define model
    model = CNN_1d(num_class, dropout_rate).to(device)
    summary(model, depth=3, input_size=(batch_size, 1, 3000))
    # return
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    loss = nn.CrossEntropyLoss()

    train_loader, val_loader = torch_data_loader_pickle_wo0(
        input_pickle_file, batch_size, val_rate, total_rate=total_rate)

    size = len(train_loader.dataset)
    print('train_loader size: ', size)

    max_acc = 0
    min_loss = np.inf
    for epoch in range(1, epochs+1):
        start_time = time.time()
        # set model to train mode
        model.train()

        # for batch_i, (X, Y) in enumerate(tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        for batch_i, (X, Y) in enumerate(train_loader):
            batches_done = len(train_loader) * epoch + batch_i
            # Y = Y.type(torch.LongTensor)
            X = X.to(device)
            Y = Y.to(device)
            Y_hat = model(X)
            # print(Y.size(), Y)
            # print(Y_hat.size(),Y_hat)
            
            # exit(0)
            
            l = loss(Y_hat, Y)
            l.backward()
            # Run optimizer
            optimizer.step()
            # Reset gradients
            optimizer.zero_grad()

            if batch_i % int((size//batch_size)*0.25) == 0:
                l, current = l.item(), batch_i * len(X)
                print(f"loss: {l:>7f}  [{current:>5d}/{size:>5d}]")

        correct = 0
        total = 0
        # set model to train mode
        model.eval()
        with torch.no_grad():
            running_loss = 0.0 
            total = 0 
            for batch_i, (X, Y) in enumerate(tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch}", mininterval=1)):
                X = X.to(device)
                Y = Y.to(device)
                Y_hat = model(X)
                # print(Y_hat)
                val_loss = loss(Y_hat, Y)

                total += Y.size(0)

                pred = torch.argmax(Y_hat, dim=1)
                # labels = torch.argmax(Y, dim=1)
                labels = Y
                correct += (pred == labels).sum().item()

                running_loss += val_loss.item() * Y.size(0)
                cross_acc = 100 * correct / total
            
            epoch_loss = running_loss / total
            epoch_acc = correct/total * 100
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                print('='*20,'save the best model based on lowest loss','='*20)
                torch.save(model.state_dict(), model_out_path)

                max_acc = cross_acc

        scheduler.step()
        print('last lr:', scheduler.get_last_lr())
        print(
            f'\nAccuracy of the model on the {total} val sets: {epoch_acc:>.3f}%    loss: {epoch_loss:>.6f}')
        end_time = time.time()  # 记录epoch结束的时间
        epoch_duration = end_time - start_time  # 计算epoch的持续时间
        print(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds \n\n")

    return


if __name__ == "__main__":
    num_gpu_devices = torch.cuda.device_count()
    print("可用的GPU设备数量:", num_gpu_devices)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    num_class = 96
    
    ################
    # load configs #
    ################
    total_rate = 1.0
    val_rate = 0.05
    dropout_rate = 0.5
    batch_size = 1024
    epochs=40
    learning_rate=0.001
    weight_decay=0.0001

    print('=' * 95)
    print('num_class:%d' % num_class)
    print('batch_size:%d' % batch_size)
    print('epochs:%d' % epochs)
    print('learning_rate:%f' % learning_rate)
    print('weight_decay:%f' % weight_decay)
    print('dropout_rate:%f' % dropout_rate)
    print('Dataset: %s' % input_pickle_file)
    print('=' * 95)

    setup_seed(114514)

    current_datetime = datetime.datetime.now()
    print("Training start from: ", current_datetime)
    
    train(num_class,batch_size,epochs,learning_rate,weight_decay,dropout_rate,total_rate, val_rate, device)
    
    current_datetime = datetime.datetime.now()
    print("Training end at: ", current_datetime)
