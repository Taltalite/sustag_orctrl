import torch
from torch import nn
import sys
sys.path.append('/home/lijy/workspace/')
from sustag_orctrl.model.model_CNN import CNN_1d
# from sustag.model.model_CNN_1k5 import CNN_1d
from sustag_orctrl.utils.load_data_wo0 import torch_data_loader_pickle_test
from torchinfo import summary
import math

import configparser
import os
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
from matplotlib.colors import LinearSegmentedColormap


DATA_TYPE='cnn_sustag96_wo0'


# input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/ONT_1d3000masked/train_set/barcode_all_1d3000_ONT_100_1wun.P'
# input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/seq96_1d3000masked_max4000/train_set/barcode_all_1d3000_sustag96_100_1wun.P'
input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/seq96_1d3000masked_max4000/train_set/barcode_all_1d3000_sustag96_100_wo0.P'
# input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/seq384_1d1500masked/train_set/barcode_all_1d1500_seq384_100_5wun.P'

log_dir = r'/home/lijy/workspace/sustag_orctrl/test/output/sustag96/'

def plot_heatmap(cm, outpath):    
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ['#003F43', '#2cb8b4', '#003F43'], N=128)
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=False, vmin=0.0, vmax=0.01)  # 画热力图

    x_interval = 2 
    y_interval = 2  

    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    print(x_ticks)

    ax.set_title('{DATA_TYPE} confusion matrix'.format(DATA_TYPE=DATA_TYPE), fontsize=18)  # 标题
    ax.set_xlabel('Predict', fontsize=18)  # x轴
    ax.set_ylabel('Ground truth', fontsize=18)  # y轴
    plt.savefig(outpath, dpi=300)
    
    return

def test(num_class,batch_size,val_rate, device):

    print("load model")
    model_path = r'/home/lijy/workspace/sustag_orctrl/checkpoints/CNN/'
    model_name = f"CNN_sustag96_1d3000_masked_ckpt_e40_b1024_0un.pth"

    model = CNN_1d(num_class, 0.5).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, model_name), map_location=device))

    print("load data")
    # Get data
    total_rate = 0.05
    val_rate = 0.05
    # _, test_loader = torch_data_loader_dir(
    #     input_spectrogram_dir, model_params.n_classes, batch_size, total_rate, val_rate, True)
    val_loader = torch_data_loader_pickle_test(input_pickle_file, num_class, batch_size, val_rate)

    print("evaluating...")
    _evaluate(model, val_loader, batch_size, num_class, log_dir, device, WITHOUT0=False)


def _evaluate(model, dataloader, batch_size, n_classes, log_dir, device, WITHOUT0=False):

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()  # Set model to evaluation mode
    # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    pred = []
    correct = 0
    total = 0
    sample_metrics = {}
    for batch_i, (X, Y) in enumerate(tqdm.tqdm(dataloader, desc="Validating")):
        X = X.to(device)
        Y = Y.to(device)
        with torch.no_grad():
            Y_hat = model(X)
            total += len(Y_hat)
            if WITHOUT0:
                pred.append(torch.argmax(Y_hat, dim=1).cpu().numpy() + 1)
            else:
                pred.append(torch.argmax(Y_hat, dim=1).cpu().numpy())            
            
            labels.append(Y.squeeze().cpu().numpy())

    
    test_pred = np.empty((total,), dtype=int)
    test_label = np.empty((total,), dtype=int)
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            test_pred[batch_size*i+j] = pred[i][j]
            test_label[batch_size*i+j] = labels[i][j]
    
    print(test_pred.shape, test_pred)
    print(test_label.shape, test_label)

    # for _ in range(test_pred.shape[0]):
    #     if test_pred[_] == test_label[_]:
    #         correct += 1

    acc_score = metrics.accuracy_score(test_label, test_pred)
    sample_metrics['accuracy'] = acc_score

    f1_score = metrics.f1_score(test_label, test_pred, average='weighted')
    sample_metrics['f1_score_weighted'] = f1_score
    
    recall = metrics.recall_score(test_label, test_pred,  average='weighted')
    sample_metrics['recall'] = recall

    precision = metrics.precision_score(test_label, test_pred,  average='weighted')
    sample_metrics['precision'] = precision

    if not WITHOUT0:
        bi_metrics = biclass_metrics(test_label, test_pred)
        sample_metrics['biclass_metrics'] = bi_metrics

    test_label_without_un, test_pred_without_un = remove_unclassified(test_label,test_pred)
    f1_score_without_un = metrics.f1_score(test_label_without_un, test_pred_without_un, average='weighted')
    sample_metrics['f1_score_weighted_without_un'] = f1_score_without_un
    acc_without_un = metrics.accuracy_score(test_label_without_un, test_pred_without_un)
    sample_metrics['acc_score_without_un'] = acc_without_un
    precision_weighted_without_un = metrics.precision_score(test_label_without_un, test_pred_without_un, average='weighted')
    sample_metrics['precision_weighted_without_un'] = precision_weighted_without_un
    recall_weighted_without_un = metrics.recall_score(test_label_without_un, test_pred_without_un,  average='weighted')
    sample_metrics['recall_weighted_without_un'] = recall_weighted_without_un

    with open('%s/{DATA_TYPE}_metrics_truetest.txt'.format(DATA_TYPE=DATA_TYPE.lower()) % log_dir, 'w') as file:
        # print(sample_metrics, file=file)
        for key, value in sample_metrics.items():
            print("{}: {}".format(key, value), file=file)


    conf_mat = metrics.confusion_matrix(test_label, test_pred)
    # print(conf_mat)

    conf_mat_sum = conf_mat.sum(axis=0)[:, np.newaxis]
    conf_mat_sum[conf_mat_sum == 0] = 1
    # print(conf_mat_sum)
    cm_normalized = conf_mat.astype('float') / conf_mat_sum

    print(cm_normalized.shape)

    np.save('%s/{DATA_TYPE}_confusion_matrix_nonorm.npy'.format(DATA_TYPE=DATA_TYPE.lower()) % log_dir, conf_mat)

    plot_heatmap(cm_normalized, '%s/{DATA_TYPE}_test_cm_0.1_test_truetest.png'.format(DATA_TYPE=DATA_TYPE.lower()) % log_dir)

    return


def biclass_metrics(test_label: np.array, test_pred: np.array):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    bi_metrics = {}
    for i in range(len(test_label)):
        if test_label[i] == 0 and test_pred[i] == 0:
            TP += 1
        elif test_label[i] == 0 and test_pred[i] != 0:
            FN += 1
        elif test_label[i] != 0 and test_pred[i] == 0:
            FP += 1
    print(TP, FN, FP, TN)
    TN = len(test_label) - TP - FN - FP
    bi_metrics['accuracy'] = (TP + TN) / (len(test_label))
    bi_metrics['precision'] = (TP) / (TP+FP)
    bi_metrics['recall'] = (TP) / (TP+FN)
    bi_metrics['f1_score'] = 2*bi_metrics['precision'] * \
        bi_metrics['recall'] / (bi_metrics['precision'] + bi_metrics['recall'])
    
    return bi_metrics

def remove_unclassified(test_label: np.array, test_pred: np.array):
    test_label_without_un = []
    test_pred_without_un = []
    for i in range(len(test_label)):
        if test_label[i] != 0:
            test_label_without_un.append(test_label[i])
            test_pred_without_un.append(test_pred[i])
    
    test_label_without_un = np.array(test_label_without_un, dtype=int)
    test_pred_without_un= np.array(test_pred_without_un, dtype=int)

    return test_label_without_un, test_pred_without_un


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # num_gpu_devices = torch.cuda.device_count()
    # print("Available GPU nums:", num_gpu_devices)
    # if torch.cuda.is_available():
    #     target_device = 0
    #     torch.cuda.set_device(target_device)

    #     device = torch.cuda.current_device()
    #     total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)  # 总显存大小（MB）
    #     allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # 已分配的显存大小（MB）
    #     reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)  # 保留的显存大小（MB）

    #     print("GPU device index:", device)
    #     print("Memory:", total_memory)
    #     print("Allocated memory:", allocated_memory)
    #     print("reserved memory:", reserved_memory)

    # else:
    #     print("No available GPU, using CPU")
    #     device = torch.device("cpu")
    ################
    # load configs #
    ################
    num_class = 96
    batch_size = 512
    val_rate = 1.0


    test(num_class,batch_size,val_rate,device)
