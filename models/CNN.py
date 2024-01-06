import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels=10, out_channels=10):
        super(CNN, self).__init__()
        self.seq = torch.nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(),
        )

    def forward(self, X):
        return self.seq(X)
