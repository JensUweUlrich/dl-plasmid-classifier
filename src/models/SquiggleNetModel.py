"""
This model is from the SquiggleNet project, see https://github.com/welch-lab/SquiggleNet/blob/master/model.py
"""

from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import balanced_accuracy_score

def conv3(in_channel, out_channel, stride=1, padding=1, groups=1):
    return nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, bias=False,
                     dilation=padding, groups=groups)


def conv1(in_channel, out_channel, stride=1, padding=0):
    return nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, padding=padding, bias=False)


def bcnorm(channel):
    return nn.BatchNorm1d(channel)


class Bottleneck(nn.Module):
    expansion = 1.5

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1(in_channel, in_channel)
        self.bn1 = bcnorm(in_channel)
        self.conv2 = conv3(in_channel, in_channel, stride)
        self.bn2 = bcnorm(in_channel)
        self.conv3 = conv1(in_channel, out_channel)
        self.bn3 = bcnorm(out_channel)
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


class SquiggleNet(nn.Module):
    def __init__(self, block, layers):
        super(SquiggleNet, self).__init__()
        self.chan1 = 20

        # first block
        self.conv1 = nn.Conv1d(1, 20, 19, padding=5, stride=3)
        self.bn1 = bcnorm(self.chan1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2, padding=1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Note JUU 17/10/2023 
        # Only one output neuron for binary classification with
        # binary cross entropy loss function
        self.fc = nn.Linear(67, 1)

        self.layer1 = self._make_layer(block, 20, layers[0])
        self.layer2 = self._make_layer(block, 30, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 45, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 67, layers[3], stride=2)

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
                conv1(self.chan1, channels, stride),
                bcnorm(channels),
            )

        layers = [block(self.chan1, channels, stride, downsample)]
        if stride != 1 or self.chan1 != channels:
            self.chan1 = channels
        for _ in range(1, blocks):
            layers.append(block(self.chan1, channels))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class SquiggleNetLightning(pl.LightningModule):

    def __init__(self, train_pos_weight, val_pos_weight,
                 learning_rate=0.001, batch_size=100, num_blocks=2, num_layers=4):
        super(SquiggleNetLightning, self).__init__()
        #self.save_hyperparameters()
        self.chan1 = 20
        block = Bottleneck
        #print(config)

        self.lr = learning_rate
        self.batch_size = batch_size
        self.train_criterion = nn.BCEWithLogitsLoss(pos_weight=train_pos_weight)
        self.val_criterion = nn.BCEWithLogitsLoss(pos_weight=val_pos_weight)

        # first block
        self.conv1 = nn.Conv1d(1, 20, 19, padding=5, stride=3)
        self.bn1 = bcnorm(self.chan1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2, padding=1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        

        layers = []
        layers.append(self._make_layer(block, channels=20, blocks=num_blocks))
        channels = 20
        for i in range(1, num_layers):
            channels = int(float(channels) * 1.5)
            layers.append(self._make_layer(block, channels, blocks=num_blocks, stride=2))

        self.hidden_layers = nn.Sequential(*layers)
        # Note JUU 17/10/2023 
        # Only one output neuron for binary classification with
        # binary cross entropy loss function
        self.fc = nn.Linear(channels, 1)

        #self.layer1 = self._make_layer(block, 20, layers[0])
        #self.layer2 = self._make_layer(block, 30, layers[1], stride=2)
        #self.layer3 = self._make_layer(block, 45, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 67, layers[3], stride=2)

        self.eval_loss = []
        self.eval_accuracy = []

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
                conv1(self.chan1, channels, stride),
                bcnorm(channels),
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

        x = self.hidden_layers(x)

        #x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

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
        self.eval_loss.append(val_loss)
        self.eval_accuracy.append(val_acc)
        #return val_loss
        return {"val_loss": val_loss, "val_bal_acc": val_acc}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, sync_dist=True)
        self.eval_loss.clear()
        self.eval_accuracy.clear()