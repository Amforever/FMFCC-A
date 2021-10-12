import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


###ResNet残差网络
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=2):
        super().__init__()

        self.in_channels = 64

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.logsoftmax = nn.LogSoftmax(dim=1)
        # self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.shape.__len__()==4:
            batch_size, chnal, h, w = x.size()         # 使用滤波特征
            x = x.view(batch_size, chnal, h, w)  # 对于cnn模型，直接将conv1的channel的个数设置为8即可，
            x = self.conv1_8(x)
        else:
            batch_size,  h, w = x.size()             # # 使用原始的mfcc等特征
            x = x.view(batch_size, 1, h, w)
            x = self.conv1_1(x)
        output=x
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        # output = self.softmax(output)

        return output

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])









'''Huaguang End-to-End model'''
# ResNet-style module
class RSM1D(nn.Module):
    def __init__(self, channels_in=None, channels_out=None):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.conv1 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(channels_out)
        self.bn2 = nn.BatchNorm1d(channels_out)
        self.bn3 = nn.BatchNorm1d(channels_out)

        self.nin = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=1)

    def forward(self, xx):
        yy = F.relu(self.bn1(self.conv1(xx)))
        yy = F.relu(self.bn2(self.conv2(yy)))
        yy = self.conv3(yy)
        xx = self.nin(xx)

        xx = self.bn3(xx + yy)
        xx = F.relu(xx)
        return xx


class SSDNet1D(nn.Module):  # Res-TSSDNet
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)

        self.RSM1 = RSM1D(channels_in=16, channels_out=32)
        self.RSM2 = RSM1D(channels_in=32, channels_out=64)
        self.RSM3 = RSM1D(channels_in=64, channels_out=128)
        self.RSM4 = RSM1D(channels_in=128, channels_out=128)

        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=4)

        # stacked ResNet-Style Modules
        x = self.RSM1(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM2(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM3(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM4(x)
        # x = F.max_pool1d(x, kernel_size=x.shape[-1])
        x = F.max_pool1d(x, kernel_size=125)  #  375:6s  250：4s   125：2s  62

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


def HGRes_TSSDNet():
    """ return a ResNet 101 object
    """
    return SSDNet1D()


