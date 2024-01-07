import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_dim=1024, output_dim=10, input_dim=28*28, depth=1):
        super(MLP, self).__init__()
        lin1 = nn.Linear(input_dim, hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        lin3 = nn.Linear(hidden_dim, output_dim)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, x):
        out = x.view(x.shape[0], 10, 28*28).sum(dim=1)
        out = self._main(out)
        return out

