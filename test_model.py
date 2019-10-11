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
    def __init__(self, in_channel, num_classes, lr, batch_size, weight_decay, num_epoches):
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