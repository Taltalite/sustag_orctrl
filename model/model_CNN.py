import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

class CNN_1d(nn.Module):

    def __init__(self, num_class, dropout_rate):

        super(CNN_1d, self).__init__()

        O_1, O_2, O_3, O_4, O_5 = 64, 128, 256, 512, 1024
        K_1, K_2, K_3, K_4, K_5 = 15, 8, 6, 4, 2
        KP_1, KP_2, KP_3, KP_4, KP_5 = 6, 3, 2, 2, 1
        FN_1, FN_2 = 4096, 1024

        self.conv1 = nn.Sequential(nn.Conv1d(1, O_1, K_1, stride=1), nn.ReLU(), nn.AvgPool1d(KP_1))
        self.conv1_bn = nn.BatchNorm1d(O_1)

        self.conv2 = nn.Sequential(nn.Conv1d(O_1, O_2, K_2), nn.ReLU(), nn.AvgPool1d(KP_2))
        self.conv2_bn = nn.BatchNorm1d(O_2)

        self.conv3 = nn.Sequential(nn.Conv1d(O_2, O_3, K_3), nn.ReLU(), nn.AvgPool1d(KP_3))
        self.conv3_bn = nn.BatchNorm1d(O_3)

        self.conv4 = nn.Sequential(nn.Conv1d(O_3, O_4, K_4), nn.ReLU(), nn.AvgPool1d(KP_4))
        self.conv4_bn = nn.BatchNorm1d(O_4)

        self.conv5 = nn.Sequential(nn.Conv1d(O_4, O_5, K_5), nn.ReLU(), nn.AvgPool1d(KP_5))
        self.conv5_bn = nn.BatchNorm1d(O_5)

        # not used, but is in the model file for some reason
        self.gru1 = nn.GRU(input_size=92160, hidden_size=10, num_layers=1)

        self.fc1 = nn.Linear(37888, FN_1, nn.Dropout(dropout_rate))
        self.fc1_bn = nn.BatchNorm1d(FN_1)

        self.fc2 = nn.Linear(FN_1, FN_2, nn.Dropout(dropout_rate))
        self.fc2_bn = nn.BatchNorm1d(FN_2)

        self.fc3 = nn.Linear(FN_2, num_class)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_bn(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_bn(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3_bn(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.relu(self.conv4_bn(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.relu(self.conv5_bn(x))
        x = x.view(len(x), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1_bn(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2_bn(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    batch_size = 256
    num_class = 96
    # Define model
    model = CNN_1d(num_class, 0.5).to(device)
    summary(model, depth=6, input_size=(batch_size,1, 3000))