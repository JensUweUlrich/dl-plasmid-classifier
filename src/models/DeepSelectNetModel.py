"""
This model is from the DeepSelectNet project, see https://github.com/AnjanaSenanayake/DeepSelectNet/blob/master/core/ResNet.py
"""

import torch

from torch import nn
import lightning.pytorch as pl
from sklearn.metrics import balanced_accuracy_score


class DSBottleneck(nn.Module):
    expansion = 1.5

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(DSBottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, in_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channel)
        self.conv2 = nn.Conv1d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, bias=False,
                     dilation=1, groups=1)
        self.bn2 = nn.BatchNorm1d(in_channel)
        self.conv3 = nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 

class DeepSelectNet(pl.LightningModule):

    def __init__(self, block, config):
        super(DeepSelectNet, self).__init__()
        #self.save_hyperparameters()

        self.lr = config['learning_rate']
        self.batch_size = config['batch_size']
        self.train_criterion = nn.BCEWithLogitsLoss(pos_weight=config['train_pos_weight'])
        self.val_criterion = nn.BCEWithLogitsLoss(pos_weight=config['val_pos_weight'])

        self.chan1 = 20
        # first block
        self.conv1 = nn.Conv1d(1, 20, 19, padding=5, stride=3)
        self.bn1 = nn.BatchNorm1d(20)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2, padding=1, stride=2)

        #self.dropout1 = nn.Dropout(0.1)
        #self.dropout2 = nn.Dropout(0.1)
        #self.dropout3 = nn.Dropout(0.1)
        #self.dropout4 = nn.Dropout(0.1)


        layers = []
        layers.append(self._make_layer(block, channels=20, blocks=config['num_blocks']))
        channels = 20
        for i in range(1, config['num_layers']):
            layers.append(nn.Dropout(config['dropout']))
            channels = int(float(channels) * 1.5)
            layers.append(self._make_layer(block, channels, blocks=config['num_blocks'], stride=2))

        self.hidden_layers = nn.Sequential(*layers)

        #self.layer1 = self._make_layer(block, 20, layers[0])
        #self.layer2 = self._make_layer(block, 30, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 45, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 67, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.noise_layer = GaussianNoise(10)

        # Note JUU 17/10/2023
        # Only one output neuron for binary classification with
        # binary cross entropy loss function
        # no final sigmoid layer needed because BCEwithLogitsLoss includes sigmoid layer
        # and is numerically more stable
        # https://medium.com/dejunhuang/learning-day-57-practical-5-loss-function-crossentropyloss-vs-bceloss-in-pytorch-softmax-vs-bd866c8a0d23
        self.fc = nn.Linear(channels, 1)
        #self.sigmoid = nn.Sigmoid()

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.chan1 != channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.chan1, channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm1d(channels),
            )

        layers = [block(self.chan1, channels, stride, downsample)]
        if stride != 1 or self.chan1 != channels:
            self.chan1 = channels
        for _ in range(1, blocks):
            layers.append(block(self.chan1, channels))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        #x = self.dropout1(x)
        #x = self.layer1(x)
        #x = self.dropout2(x)
        #x = self.layer2(x)
        #x = self.dropout3(x)
        #x = self.layer3(x)
        #x = self.dropout4(x)
        #x = self.layer4(x)

        x = self.hidden_layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.noise_layer(x)
        x = self.fc(x)

        # see above
        #x = self.sigmoid(x)

        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
     
    def training_step(self, train_batch, train_batch_idx):
        data, labels, ids = train_batch
        train_labels = labels.to(torch.float)
        outputs_train = self.forward(data)
        train_loss = self.train_criterion(outputs_train, labels.unsqueeze(1))
        pred_labels = (outputs_train >= 0.5).type(torch.float)
        train_acc = (train_labels == pred_labels).float().mean().item()
        self.log('train_loss', train_loss, batch_size=self.batch_size, )
        self.log('train_acc', train_acc, batch_size=self.batch_size)
        return train_loss

    def validation_step(self, val_batch, batch_idx):
        data, labels, val_ids = val_batch
        val_outputs = self.forward(data)
        val_loss = self.val_criterion(val_outputs, labels.unsqueeze(1).to(torch.float))
        predicted_labels = (val_outputs >= 0.5).type(torch.long).cpu().data.numpy()
        val_labels = labels.to(torch.long).cpu().numpy()
        val_acc = torch.tensor(balanced_accuracy_score(val_labels, predicted_labels))
        self.log('val_loss', val_loss, batch_size=self.batch_size)
        self.log('val_bal_acc', val_acc, batch_size=self.batch_size)
        return val_loss
