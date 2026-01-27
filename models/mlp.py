# This code is adapted and modified from the Power-Flow-Balancing-with-Decentralized-Graph-Neural-Networks
# repository in order to work with PyTorch and fit the data/structure requirements of the PathDistFlow project.
#
# Original code: https://github.com/JonasBergHansen/Power-Flow-Balancing-with-Decentralized-Graph-Neural-Networks (GitHub: JonasBergHansen)
# Original paper: https://doi.org/10.1109/TPWRS.2022.3195301.

import torch
import torch.nn as nn

class GlobalMLP(nn.Module):
    MAX_NODE_FEATURES = 129 * 5
    MAX_EDGE_FEATURES = 129 * 4
    MAX_OUTPUT_FEATURES = 129 * 2

    def __init__(self, node_dim_in=MAX_NODE_FEATURES, edge_dim_in=MAX_EDGE_FEATURES, out_dim=MAX_OUTPUT_FEATURES, hidden_dim=256, dropout=0.0):
        super().__init__()

        # Activations
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.dropout = dropout

        # Pre-processing layers (64 units as per Hansen et al.)
        self.bus_preprocess = nn.Linear(node_dim_in, hidden_dim // 2)
        self.edge_preprocess = nn.Linear(edge_dim_in, hidden_dim // 2)

        # Core MLP layers (128 units as per Hansen et al.)
        # Input is 64 (bus) + 64 (edge) = 128
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layer
        self.readout = nn.Linear(hidden_dim, out_dim)

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
        """
        x: Flattened node features [Batch, MAX_NODE_FEATURES + MAX_EDGE_FEATURES]
        """
        # Input cleaning
        x = torch.nan_to_num(x, nan=0.0)

        # Pre-processing
        bus_out = self.leakyReLU(self.bus_preprocess(x[:, :self.MAX_NODE_FEATURES]))
        edge_out = self.leakyReLU(self.edge_preprocess(x[:, self.MAX_NODE_FEATURES:self.MAX_NODE_FEATURES + self.MAX_EDGE_FEATURES]))

        # Concatenation
        out = torch.cat([bus_out, edge_out], dim=1)

        # Processing
        out = self.leakyReLU(self.dense1(out))
        out = nn.functional.dropout(out, self.dropout, training=self.training) # Added dropout parameter
        out = self.leakyReLU(self.dense2(out))
        out = self.readout(out)

        # Dont need to zero out padded nodes, because ignore them in loss calculation and metrics
        return out


class CustomNormedMLP(nn.Module):
    """A custom MLP model (not from Hansen et al.) for power flow prediction."""
    MAX_NODE_FEATURES = 129*5
    MAX_EDGE_FEATURES = 129*4
    MAX_OUTPUT_FEATURES = 129*2
    def __init__(self, input_dim=MAX_NODE_FEATURES + MAX_EDGE_FEATURES, output_dim=MAX_OUTPUT_FEATURES, hidden_dim=256):
        super().__init__()
        # Computational Layers
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

        # Regularization layers
        self.norms1 = nn.BatchNorm1d(hidden_dim)
        self.norms2 = nn.BatchNorm1d(hidden_dim)

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
