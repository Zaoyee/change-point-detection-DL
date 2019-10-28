import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim

def conv1x3(in_channel, out_channel, stride=1):
    return nn.Conv1d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, 1, 1,bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)

        self.conv2 = conv1x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)

        self.conv3 = nn.Conv1d(out_channel, in_channel, 1, 1,bias=False)
        self.bn3 = nn.BatchNorm1d(in_channel)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out, True)
        return F.relu(x+out, True)

class resnet(nn.Module):
    def __init__(self, in_channel, num_classes, lr, batch_size, weight_decay, num_epoches, **kwargs):
        super(resnet, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.num_epoches = num_epoches

        self.layer1 = nn.Conv1d(1, 64, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.block1 = residual_block(64, 128)
        self.block2 = residual_block(64, 128)

        self.classifier = nn.Linear(6400, 1024)
        self.classifier2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = x.float()
        x = self.bn(self.layer1(x))
        x = F.relu(x, True)
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x.view(x.size(0), -1))
        x = F.relu(x, True)
        x = self.classifier2(x)
        return x

class resnet2(nn.Module):
    def __init__(self, in_channel, num_classes, lr, batch_size, weight_decay, num_epoches, **kwargs):
        super(resnet, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.num_epoches = num_epoches

        self.layer1 = nn.Conv1d(1, 64, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.block1 = residual_block(64, 128)
        self.block2 = residual_block(64, 128)
        self.block3 = residual_block(64, 128)

        self.classifier = nn.Linear(6400, 1024)
        self.classifier2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = x.float()
        x = self.bn(self.layer1(x))
        x = F.relu(x, True)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x.view(x.size(0), -1))
        x = F.relu(x, True)
        x = self.classifier2(x)
        return x


class simple1dcnn(nn.Module):
    def __init__(self, lr, batch_size, weight_decay, num_epoches, **kwargs):
        super(simple1dcnn, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.num_epoches = num_epoches
        # 1 x 100
        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv1d(in_channels=1, out_channels=100, kernel_size=11, stride=1))
        layer1.add_module('bn1', nn.BatchNorm1d(100))
        layer1.add_module('relu1', nn.ReLU(True))
        # 200 x 90
        layer1.add_module('pool1', nn.MaxPool1d(2, 2))
        self.layer1 = layer1
        # 200 x 45

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv1d(in_channels=100, out_channels=100, kernel_size=10, stride=1))
        layer2.add_module('bn2', nn.BatchNorm1d(100))
        layer2.add_module('relu2', nn.ReLU(True))
        # 200 x 36
        layer2.add_module('pool2', nn.MaxPool1d(2, 2))
        self.layer2 = layer2
        # 100 x 18

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv1d(in_channels=100, out_channels=100, kernel_size=9, stride=1))
        layer3.add_module('bn3', nn.BatchNorm1d(100))
        layer3.add_module('relu3', nn.ReLU(True))
        # 100 x 10
        layer3.add_module('pool3', nn.MaxPool1d(2, 2))
        self.layer3 = layer3
        # 100 x 5

        self.layer4 = nn.Sequential()
        self.layer4.add_module('full_connected1', nn.Linear(500, 256))
        self.layer4.add_module('fc_relu1', nn.ReLU(True))
        self.layer4.add_module('full_connected2', nn.Linear(256, 124))
        self.layer4.add_module('fc_relu2', nn.ReLU(True))
        self.layer4.add_module('full_connected3', nn.Linear(124, 1))

    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        return x
