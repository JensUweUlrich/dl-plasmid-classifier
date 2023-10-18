"""
This model is from the DeepSelectNet project, see https://github.com/AnjanaSenanayake/DeepSelectNet/blob/master/core/ResNet.py
"""

import torch

from torch import nn


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

class DeepSelectNet(nn.Module):

    def __init__(self, block, layers):
        super(DeepSelectNet, self).__init__()
        self.chan1 = 20
        # first block
        self.conv1 = nn.Conv1d(1, 20, 19, padding=5, stride=3)
        self.bn1 = nn.BatchNorm1d(20)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2, padding=1, stride=2)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)

        self.layer1 = self._make_layer(block, 20, layers[0])
        self.layer2 = self._make_layer(block, 30, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 45, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 67, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.noise_layer = GaussianNoise(10)

        # Note JUU 17/10/2023
        # Only one output neuron for binary classification with
        # binary cross entropy loss function
        # no final sigmoid layer needed because BCEwithLogitsLoss includes sigmoid layer
        # and is numerically more stable
        # https://medium.com/dejunhuang/learning-day-57-practical-5-loss-function-crossentropyloss-vs-bceloss-in-pytorch-softmax-vs-bd866c8a0d23
        self.fc = nn.Linear(67, 1)
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

        x = self.dropout1(x)
        x = self.layer1(x)
        x = self.dropout2(x)
        x = self.layer2(x)
        x = self.dropout3(x)
        x = self.layer3(x)
        x = self.dropout4(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.noise_layer(x)
        x = self.fc(x)

        # see above
        #x = self.sigmoid(x)

        return x

#    def conv_pass_1(self, x, n_feature_maps=20):
#        x1 = nn.Conv1d(1, n_feature_maps, kernel_size=1, strides=1, padding=padding, bias=False,
#                     dilation=padding, groups=groups)
#        x1 = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='causal', strides=1)(x)
#        x1 = tf.keras.layers.BatchNormalization()(x1)
#        x1 = tf.keras.layers.Activation('relu')(x1)
#        return x1

#    def conv_pass_2(self, x, n_feature_maps=20, strides=1):
#        x1 = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='causal', strides=strides)(x)
#        x1 = tf.keras.layers.BatchNormalization()(x1)
#        x1 = tf.keras.layers.Activation('relu')(x1)
#        return x1

#    def conv_pass_3(self, x, n_feature_maps=20):
#        x1 = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='causal', strides=1)(x)
#        x1 = tf.keras.layers.BatchNormalization()(x1)
#        x1 = tf.keras.layers.Activation('relu')(x1)
#        return x1

#    def make_layer(self, x, filters, blocks, strides=1):
#        filter_1 = 20
#        down_sample = None

#        if strides != 1 or filter_1 != filters:
#            down_sample = True

#        x = self.conv_pass_1(x, filters)
#        x = self.conv_pass_2(x, filters, strides=strides)
#        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding='causal', strides=1)(x)
#        x = tf.keras.layers.BatchNormalization()(x)
#        if down_sample:
#            x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=strides)(x)
#            x = tf.keras.layers.BatchNormalization()(x)
#        x = tf.keras.layers.Activation('relu')(x)

#        if down_sample:
#            filter_1 = filters

#        for _ in range(1, blocks):
#            x = self.conv_pass_1(x, filters)
#            x = self.conv_pass_2(x, filters)
#            x = self.conv_pass_3(x, filters)
#        return x

#    def build_model(self, input_shape, nb_classes, is_train):

#        input_layer = tf.keras.layers.Input(input_shape)

        # BLOCK 1
#        x = tf.keras.layers.Conv1D(filters=20, kernel_size=19, padding='causal', strides=3)(input_layer)
#        x = tf.keras.layers.BatchNormalization()(x)
#        x = tf.keras.layers.Activation('relu')(x)
#        x = tf.keras.layers.MaxPooling1D(2, padding='valid', strides=2)(x)

        # LAYERS
#        x = tf.keras.layers.Dropout(0.10)(x)
#        x = self.make_layer(x, filters=20, blocks=2)
#        x = tf.keras.layers.Dropout(0.10)(x)
#        x = self.make_layer(x, filters=30, blocks=2, strides=2)
#        x = tf.keras.layers.Dropout(0.10)(x)
#        x = self.make_layer(x, filters=45, blocks=2, strides=2)
#        x = tf.keras.layers.Dropout(0.10)(x)
#        x = self.make_layer(x, filters=67, blocks=2, strides=2)

        # FINAL
#        x = tf.keras.layers.AveragePooling1D(1)(x)

        # how to add this layer with Pytorch?
        
#        gap_layer = tf.keras.layers.Flatten()(x)

#        noise_layer = K.zeros_like(gap_layer)
#        noise_layer = tf.keras.layers.GaussianNoise(10)(noise_layer)

#        if is_train:
#            x = tf.keras.layers.Lambda(lambda z: tf.concat(z, axis=0))([gap_layer, noise_layer])
#        else:
#            x = gap_layer

#        output_layer = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(x)

#        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

#        return model