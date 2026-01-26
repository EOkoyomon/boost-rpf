import torch
import torch.nn as nn

class NormedMLP(nn.Module):
    MAX_NODE_FEATURES = 129*5
    MAX_EDGE_FEATURES = 129*4
    MAX_OUTPUT_FEATURES = 129*2
    def __init__(self, input_dim=MAX_NODE_FEATURES + MAX_EDGE_FEATURES, output_dim=MAX_OUTPUT_FEATURES):
        super().__init__()
        # Computational Layers
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, output_dim)

        # Regularization layers
        self.norms1 = nn.BatchNorm1d(256)
        self.norms2 = nn.BatchNorm1d(256)

        # Activations
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
    
    def use_physics_loss(self):
        return False

    def is_supervised(self):
        return True

    def is_complex(self):
        return False

    def is_analytical(self):
        return False

    def is_tabular(self):
        return True

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0)
        x = self.leakyReLU(self.norms1(self.layer1(x)))
        x = self.leakyReLU(self.norms2(self.layer2(x)))
        x = self.layer3(x)
        return x
