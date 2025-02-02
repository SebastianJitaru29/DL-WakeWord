import torch
import torch.nn as nn

class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, layer_dim=32):
        super(FullyConnectedNN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, layer_dim),
            nn.LayerNorm(layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, layer_dim),
            nn.LayerNorm(layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fcn(x)
