# Original CNN model for MNIST Classification
import torch
import torch.nn as nn
import torch.nn.functional as F


class OCNN(nn.Module):
    def __init__(self):
        super(OCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 输入通道为1，输出通道为32，卷积核为3x3
        self.pool = nn.MaxPool2d(2, 2)  # 池化层，2x2
        self.conv2 = nn.Conv2d(32, 64, 3)  # 第二个卷积层
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # 全连接层
        self.fc2 = nn.Linear(128, 10)  # 输出层，10个类别

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # 展平所有维度，除了批处理维度
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
