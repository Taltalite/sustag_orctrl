import torch
from torch import nn
import sys
sys.path.append('/home/lijy/workspace/')
from sustag_orctrl.model.model_ORCtrL import CNN_LSTM_Backbone, OPRNet, SelectiveLoss
from collections import OrderedDict
from sustag.utils.load_data import torch_data_loader_pickle
from torchinfo import summary
import numpy as np
import os
import time
import datetime


# input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/Porcupine_1d3000masked_max4000/train_set/barcode_all_1d3000_porcupine_100_1wun.P'
# input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/seq384_1d3000masked/train_set/barcode_all_1d3000_seq384_100_5wun.P'
input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/seq96_1d3000masked_max4000/train_set/barcode_all_1d3000_sustag96_100_4kun.P'
# input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/ONT_1d3000masked_max4000/train_set/barcode_all_1d3000_ONT_100_1wun.P'


def model_summary(model):
    summary(model, depth=5, input_size=(batch_size, 1, 3000))
    return model

def selective_coverage(out_select, threshold=0.5):
    # Convert predictions to binary decisions based on the threshold
    g = (out_select > threshold).float()
    
    # Calculate the mean of g
    coverage_value = g.mean()
    
    return coverage_value

def selective_acc(out_class,  out_select, y_true, threshold=0.5):
    # Convert predictions to binary decisions based on the threshold
    g = (out_select > threshold).float()
    g = g.squeeze()
    # Calculate the number of correct predictions where g is 1
    correct_predictions = g * (y_true == torch.argmax(out_class, dim=-1)).float()
    
    # Calculate the selective accuracy
    selective_accuracy = correct_predictions.sum() / g.sum()
    
    return selective_accuracy

def train(num_class: int, hidden_dim: int, batch_size: int, epochs: int, learning_rate: float,
          weight_decay: float, dropout_rate:float, val_rate:float, current_device):
    
    
    features = CNN_LSTM_Backbone(num_class, dropout_rate, hidden_dim).to(current_device)
    
    model = OPRNet(features, 79 * hidden_dim, num_class, dropout_rate=dropout_rate).to(current_device)    
    model_summary(model)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # loss
    coverage = 0.95
    base_loss = torch.nn.CrossEntropyLoss(reduction='none')
    SelectiveCELoss = SelectiveLoss(base_loss, coverage=coverage,num_classes=num_class, lm=32)
    criterion = torch.nn.CrossEntropyLoss()

    
    total_rate = 1.0
    train_loader, val_loader = torch_data_loader_pickle(
            input_pickle_file, num_class, batch_size, val_rate,total_rate=total_rate, is_shuffle=True, silence=False)
    size = len(train_loader.dataset)
    print('train_loader size: ', size)
    
    ckpt_path = '../checkpoints/ORCtrL/'
    os.makedirs(ckpt_path, exist_ok=True)
    model_out_path = f"{ckpt_path}/ORCtrL_sustag96_e{epochs}_b{batch_size}_t{total_rate:>.2f}_c{coverage:>.2f}_4kun.pth"
    print('model save path: ', model_out_path)
    
    max_acc = 0
    min_loss = np.inf
    ALPHA = 0.7
    for epoch in range(1, epochs+1):
        start_time = time.time()
        
        # train_metric_dict = MetricDict()
        # val_metric_dict = MetricDict()
        
        model.train()
        
        for batch_i, (X, Y) in enumerate(train_loader):
            X = X.to(current_device)
            Y = Y.to(current_device)
            out_class, out_select, out_aux = model(X)
            
            loss_dict = OrderedDict()
            selective_loss = SelectiveCELoss(out_class, out_select, Y)
            selective_loss *= ALPHA            
            loss_dict['selective_loss'] = selective_loss.detach().cpu().item()
            # compute standard cross entropy loss
            ce_loss = torch.nn.CrossEntropyLoss()(out_aux, Y)
            ce_loss *= (1.0 - ALPHA)
            loss_dict['ce_loss'] = ce_loss.detach().cpu().item()
            
            # total loss
            loss = selective_loss + ce_loss
            loss_dict['loss'] = loss.detach().cpu().item()
            
            # loss = criterion(out_class, Y)

            # backward            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch_i % int((size//batch_size)*0.4) == 0:
                l, current = loss.item(), batch_i * len(X)
                print(f"total loss: {l:>7f} selective_loss: {loss_dict['selective_loss']} ce_loss: {loss_dict['loss']} [{current:>5d}/{size:>5d}]")
        
        model.eval()
        with torch.autograd.no_grad():
            running_loss = 0.0 
            total = 0
            correct = 0
            total_selective_acc = 0.0
            total_coverage = 0.0
            for batch_i, (X, Y) in enumerate(val_loader):
                X = X.to(current_device)
                Y = Y.to(current_device)
                
                out_class, out_select, out_aux = model(X)
                
                # compute selective loss
                loss_dict = OrderedDict()
                selective_loss = SelectiveCELoss(out_class, out_select, Y)
                selective_loss *= ALPHA
                loss_dict['selective_loss'] = selective_loss.detach().cpu().item()
                # compute standard cross entropy loss
                ce_loss = torch.nn.CrossEntropyLoss()(out_aux, Y)
                ce_loss *= (1.0 - ALPHA)
                loss_dict['ce_loss'] = ce_loss.detach().cpu().item()
                
                # total loss
                loss = selective_loss + ce_loss
                loss_dict['loss'] = loss.detach().cpu().item()
                
                # loss = criterion(out_class, Y)                
                
                pred = torch.argmax(out_class, dim=1)
                # labels = torch.argmax(Y, dim=1)
                
                batch_size = X.shape[0]
                labels = Y
                correct += (pred == labels).sum().item()
                total += Y.shape[0]
                
                running_loss += loss.item() * batch_size
                total_selective_acc += selective_acc(out_class,  out_select, Y, threshold=0.5)* batch_size
                total_coverage += selective_coverage(out_select, threshold=0.5) * batch_size
            
            epoch_loss = running_loss / total
            epoch_selective_acc = total_selective_acc / total
            epoch_coverage = total_coverage / total
            epoch_acc = correct/total
            
            print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}, Selective Accuracy: {epoch_selective_acc:.4f}, Total Accuracy: {epoch_acc:.4f} Coverage: {epoch_coverage:.4f}')

            if epoch_loss < min_loss:
                min_loss = epoch_loss
                print('='*40,'save the best model based on lowest loss','='*40)
                torch.save(model.state_dict(), model_out_path)

            
        
        scheduler.step()
        print('last lr:', scheduler.get_last_lr())        
        end_time = time.time() 
        epoch_duration = end_time - start_time 
        print(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds")
        print('=' * 70)
   
    return
                
                


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

    print('=' * 100)
    print('hidden_dim:%d' % hidden_dim)
    print('num_class:%d' % num_class)
    print('batch_size:%d' % batch_size)
    print('epochs:%d' % epochs)
    print('learning_rate:%f' % learning_rate)
    print('weight_decay:%f' % weight_decay)
    print('dropout_rate:%f' % dropout_rate)
    print('Dataset: %s' % input_pickle_file)
    print('=' * 100)
    
    current_datetime = datetime.datetime.now()
    print("Training start from: ", current_datetime)

    train(num_class,hidden_dim,batch_size,epochs,learning_rate,weight_decay,dropout_rate,val_rate, device)

    current_datetime = datetime.datetime.now()
    print("Training end at: ", current_datetime)

