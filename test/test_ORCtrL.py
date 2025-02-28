import torch
from torch import nn
import sys
sys.path.append('/home/lijy/workspace/')
from sustag_orctrl.model.model_ORCtrL import CNN_LSTM_Backbone, OPRNet, SelectiveLoss
from sustag_orctrl.utils.load_data import torch_data_loader_pickle_test
from torchinfo import summary
import os
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
from matplotlib.colors import LinearSegmentedColormap

DATA_TYPE='orctrl_sustag96_c95_4kun'


input_pickle_file = r'/data/biolab-nvme-pool1/lijy/SUSTag_data/1d3000/seq96_1d3000masked_max4000/train_set/barcode_all_1d3000_sustag96_100_4kun.P'


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


def test(num_classes, dropout_rate, hidden_dim,batch_size,val_rate, device):
    print("load model")
    
    model_path = r'/home/lijy/workspace/sustag/training/checkpoints/CNNLSTMselnet/'
    model_name = f"CNNLSTMselnet_sustag96_e100_b1024_t1.00_c0.95_4kun.pth"

    features = CNN_LSTM_Backbone(num_class, dropout_rate, hidden_dim).to(device)
    
    model = OPRNet(features, 79 * hidden_dim, num_class, dropout_rate=dropout_rate).to(device)    
    # summary(model, depth=3, input_size=(batch_size, 1, 3000))
    model.load_state_dict(torch.load(os.path.join(model_path, model_name), map_location=device))

    print("load data")
    # Get data
    val_rate = 1.0
    test_loader = torch_data_loader_pickle_test(input_pickle_file, num_class, batch_size, val_rate)

    print("evaluating...")
    _evaluate(model, test_loader, batch_size, num_classes, log_dir, device)


def _evaluate(model, dataloader, batch_size, n_classes, log_dir, device):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()  # Set model to evaluation mode
    # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    pred = []
    select = []
    correct = 0
    total = 0
    sample_metrics = {}

    threshold = 0.5
    for batch_i, (X, Y) in enumerate(tqdm.tqdm(dataloader, desc="Validating")):
        X = X.to(device)
        Y = Y.to(device)
        
        with torch.no_grad():
            out_class, out_select, out_aux = model(X) 
            total += len(X)

            out_select = out_select.squeeze()
            class_preds = torch.argmax(out_class, dim=1)

            batch_pred = torch.where(out_select >= threshold, class_preds, torch.zeros_like(class_preds)).cpu().numpy()
            
            select_preds = (out_select > threshold).float().cpu().numpy()

            pred.append(batch_pred)
            select.append(select_preds)
            

            labels.append(Y.squeeze().cpu().numpy()) 

    
    test_pred = np.empty((total,), dtype=int)
    test_label = np.empty((total,), dtype=int)
    test_select = np.empty((total,), dtype=int)
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            test_pred[batch_size*i+j] = pred[i][j]
            test_label[batch_size*i+j] = labels[i][j]
            test_select[batch_size*i+j] = select[i][j]
    
    print(test_pred.shape, test_pred)
    print(test_label.shape, test_label)
    print(test_select.shape, test_select)
    # exit(0)

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
    
    # 筛选出 test_select 为 1 的样本
    selected_pred = test_pred[test_select == 1]
    selected_label = test_label[test_select == 1]

    # 计算 F1-score
    selective_f1score = metrics.f1_score(selected_label, selected_pred, average='weighted')
    selective_recall = metrics.recall_score(selected_label, selected_pred, average='weighted')
    selective_precision = metrics.precision_score(selected_label, selected_pred, average='weighted')
    actual_coverage = np.mean(test_select)   

    sample_metrics['real_coverage'] = actual_coverage
    sample_metrics['selective_f1score'] = selective_f1score
    sample_metrics['selective_recall'] = selective_recall
    sample_metrics['selective_precision'] = selective_precision
    
    selective_bi_metrics = biclass_metrics(selected_label, selected_pred)
    sample_metrics['selective_biclass_metrics'] = bi_metrics
    
    selected_label_without_un, selected_pred_without_un = remove_unclassified(selected_label, selected_pred)
    selective_f1_score_without_un = metrics.f1_score(selected_label_without_un, selected_pred_without_un, average='weighted')
    sample_metrics['selective_f1score_weighted_without_un'] = selective_f1_score_without_un
    selective_precision_weighted_without_un = metrics.precision_score(selected_label_without_un, selected_pred_without_un, average='weighted')
    sample_metrics['selective_precision_weighted_without_un'] = selective_precision_weighted_without_un
    selective_recall_weighted_without_un = metrics.recall_score(selected_label_without_un, selected_pred_without_un,  average='weighted')
    sample_metrics['selective_recall_weighted_without_un'] = selective_recall_weighted_without_un

    with open('%s/{DATA_TYPE}_metrics_truetest.txt'.format(DATA_TYPE=DATA_TYPE.lower()) % log_dir, 'w') as file:
        # print(sample_metrics, file=file)
        for key, value in sample_metrics.items():
            print("{}: {}".format(key, value), file=file)


    conf_mat = metrics.confusion_matrix(test_label, test_pred)
    select_conf_mat = metrics.confusion_matrix(selected_label, selected_pred)
    # print(conf_mat)

    conf_mat_sum = conf_mat.sum(axis=0)[:, np.newaxis]
    conf_mat_sum[conf_mat_sum == 0] = 1
    # print(conf_mat_sum)
    cm_normalized = conf_mat.astype('float') / conf_mat_sum
    
    select_conf_mat_sum = select_conf_mat.sum(axis=0)[:, np.newaxis]
    select_conf_mat_sum[select_conf_mat_sum == 0] = 1
    # print(conf_mat_sum)
    select_cm_normalized = select_conf_mat.astype('float') / select_conf_mat_sum

    print(cm_normalized.shape)

    np.save('%s/{DATA_TYPE}_confusion_matrix_select.npy'.format(DATA_TYPE=DATA_TYPE.lower()) % log_dir, select_conf_mat)
    np.save('%s/{DATA_TYPE}_confusion_matrix_nonorm.npy'.format(DATA_TYPE=DATA_TYPE.lower()) % log_dir, conf_mat)

    plot_heatmap(cm_normalized, '%s/{DATA_TYPE}_test_cm_0.1_test_truetest.png'.format(DATA_TYPE=DATA_TYPE.lower()) % log_dir)
    plot_heatmap(select_cm_normalized, '%s/{DATA_TYPE}_select_cm_0.1_test_truetest.png'.format(DATA_TYPE=DATA_TYPE.lower()) % log_dir)

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
    num_gpu_devices = torch.cuda.device_count()
    print("Available GPU nums:", num_gpu_devices)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ################
    # load configs #
    ################
    num_class = 97
    batch_size = 1024
    val_rate = 1.0
    dropout_rate=0.30
    hidden_dim=128


    test(num_class,dropout_rate,hidden_dim,batch_size,val_rate,device)
