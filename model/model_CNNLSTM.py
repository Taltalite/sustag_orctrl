import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

class CNNBlock(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(CNNBlock, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=64, kernel_size=15, stride=1),
                                   nn.ReLU(), nn.AvgPool1d(3))
        self.conv1_bn = nn.BatchNorm1d(64)

        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1),
                                   nn.ReLU(), nn.AvgPool1d(3))
        self.conv2_bn = nn.BatchNorm1d(128)

        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=6, stride=1),
                                   nn.ReLU(), nn.AvgPool1d(2))
        self.conv3_bn = nn.BatchNorm1d(256)

        self.conv4 = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4, stride=1),
                                   nn.LeakyReLU(), nn.AvgPool1d(2))
        self.conv4_bn = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_bn(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_bn(x))
        x = F.relu(self.conv3(x))
        # x = F.leaky_relu(self.conv3(x))
        x = F.relu(self.conv3_bn(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.relu(self.conv4_bn(x))

        x = self.dropout(x)

        return x


class LSTMBlock(nn.Module):
    def __init__(self, hidden_size):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, 
                            num_layers=2, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        # x = x[:, -1, :]  # 取最后一个时间步的输出
        return x



class CNN_LSTM(nn.Module):
    def __init__(self, num_class, dropout_rate, hidden_size):
        super(CNN_LSTM, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.cnn = CNNBlock(self.dropout_rate)
        self.lstm = LSTMBlock(hidden_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(79 * hidden_size),
            nn.Linear( 79 * hidden_size, 2048),
            # nn.LayerNorm(51 * hidden_size),
            # nn.Linear( 51 * hidden_size, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, self.num_class)
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=self.dropout_rate)
    
    def forward(self, x):
        #cnn takes input of shape (batch_size, channels, seq_len)
        # x = x.permute(0, 2, 1)
        cnn_out = self.cnn(x)
        # lstm takes input of shape (batch_size, seq_len, input_size)
        lstm_in = cnn_out.permute(0, 2 ,1)
        lstm_out = self.lstm(lstm_in)
        x = self.flatten(lstm_out)
        x = self.dropout(x)
        x = self.mlp(x)
        # x = self.mlp(lstm_out)
        return x





if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    batch_size = 256
    num_class = 97
    # Define model
    model = CNN_LSTM(num_class, 0.3, 128).to(device)

    summary(model, input_size=(batch_size, 1, 3000))
    

