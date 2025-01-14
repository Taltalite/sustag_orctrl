import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

class CNNBlock(nn.Module):
    def __init__(self, dropout_rate=0.2):
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
        x = self.dropout(x)
        x = F.leaky_relu(self.conv4(x))
        x = F.relu(self.conv4_bn(x))
        x = self.dropout(x)

        return x


class LSTMBlock(nn.Module):
    def __init__(self, hidden_size):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, 
                            num_layers=3, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x



class CNN_LSTM_Backbone(nn.Module):
    def __init__(self, num_class, dropout_rate, hidden_size):
        super(CNN_LSTM_Backbone, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.cnn = CNNBlock(self.dropout_rate)
        self.lstm = LSTMBlock(hidden_size)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=self.dropout_rate)
    
    def forward(self, x):
        cnn_out = self.cnn(x)
        lstm_in = cnn_out.permute(0, 2 ,1)
        features = self.lstm(lstm_in)
        # features = features[:, -1, :]
        return features

class OPRNet(torch.nn.Module):
    def __init__(self, features, dim_features:int, num_classes:int, dropout_rate=0.2, init_weights=False):
        """
        Args
            features: feature extractor network (CNN_LSTM in this work).
            dim_featues: hidden dimensions.  
        """
        super(OPRNet, self).__init__()
        self.features = features
        self.dim_features = dim_features
        self.num_classes = num_classes
        self.base_dim = 1024
        
        class LambdaLayer(nn.Module):
            def forward(self, x):
                return x / 10
        
        self.preparation = torch.nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(self.dim_features),
            nn.Linear(self.dim_features, self.base_dim),
            nn.LayerNorm(self.base_dim),
            nn.ReLU(),
        )
        
        # represented as f() in the original paper
        self.classifier = torch.nn.Sequential(    
            nn.Linear(self.base_dim, self.num_classes),
            # nn.Softmax(dim=1)
        )

        # represented as g() in the original paper
        self.selector = torch.nn.Sequential(
            nn.Linear(self.base_dim, self.base_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.base_dim),
            LambdaLayer(),
            nn.Linear(self.base_dim, 1),
            nn.Sigmoid()
        )

        # represented as h() in the original paper
        self.aux_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.base_dim, self.num_classes),
        )

        if init_weights:
            self._initialize_weights(self.classifier)
            self._initialize_weights(self.selector)
            self._initialize_weights(self.aux_classifier)

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = self.preparation(x)
        
        prediction_out = self.classifier(x)
        selection_out  = self.selector(x)
        auxiliary_out  = self.aux_classifier(x)

        return prediction_out, selection_out, auxiliary_out

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

class SelectiveLoss(torch.nn.Module):
    def __init__(self, loss_func, coverage:float, num_classes:int, lm:float=32.0):
        super(SelectiveLoss, self).__init__()
        assert 0.0 < coverage <= 1.0
        assert 0.0 < lm

        self.loss_func = loss_func
        self.coverage = coverage
        
        self.num_classes = num_classes
        self.lamda = lm

    def forward_bak(self, prediction_out, selection_out, target, threshold=0.5):

        count_above_threshold = torch.sum(selection_out > threshold).item()
        total_count = selection_out.shape[0]
        emprical_coverage = torch.tensor(count_above_threshold / total_count)
        
        # print('emprical_coverage: ', emprical_coverage)
        print(selection_out.view(-1))
        print(self.loss_func(prediction_out, target))
        # compute emprical risk (=r^)
        emprical_risk = (self.loss_func(prediction_out, target)*selection_out.view(-1)).mean()
        emprical_risk = emprical_risk / emprical_coverage
        
        # print('emprical_risk: ', emprical_risk)

        # compute penulty (=psi)
        coverage = torch.tensor([self.coverage], dtype=torch.float32, requires_grad=True, device='cuda')
        penulty = torch.max(coverage-emprical_coverage, torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device='cuda'))**2
        penulty *= self.lm
        
        # print('penulty: ', penulty, '   ', self.lm)

        selective_loss = emprical_risk + penulty

        # loss information dict 
        loss_dict={}
        loss_dict['emprical_coverage'] = emprical_coverage.detach().cpu().item()
        loss_dict['emprical_risk'] = emprical_risk.detach().cpu().item()
        loss_dict['penulty'] = penulty.detach().cpu().item()

        return selective_loss, loss_dict
    
    def forward(self, prediction_out, selection_out, target):
        # Compute the base loss without reduction
        base_loss = F.cross_entropy(prediction_out, target, reduction='none')  # Shape: [batch_size]

        # Multiply by the selection scores
        per_sample_loss = base_loss * selection_out.view(-1)

        # Compute empirical coverage
        empirical_coverage = torch.mean(selection_out)

        # Adjust the loss to account for coverage
        empirical_risk = per_sample_loss.mean() / empirical_coverage

        # Compute the penalty term
        penalty = self.lamda * torch.clamp(self.coverage - empirical_coverage, min=0) ** 2

        # Total loss
        total_loss = empirical_risk + penalty

        return total_loss

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    batch_size = 256
    num_class = 97
    dropout_rate = 0.3
    hidden_dim = 256
    features = CNN_LSTM_Backbone(num_class, dropout_rate, hidden_dim)
    
    model = OPRNet(features, 79 * 256, num_class)
    
    summary(model, input_size=(batch_size, 1, 3000))