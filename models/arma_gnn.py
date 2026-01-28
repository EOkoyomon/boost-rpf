# This code is adapted and modified from the Power-Flow-Balancing-with-Decentralized-Graph-Neural-Networks 
# repository in order to work with PyTorch and fit the data/structure requirements of the BOOST-RPF project.
#
# Original code: https://github.com/JonasBergHansen/Power-Flow-Balancing-with-Decentralized-Graph-Neural-Networks (GitHub: JonasBergHansen)
# Original paper: https://doi.org/10.1109/TPWRS.2022.3195301.

import torch
import torch.nn as nn
from torch_geometric.nn import ARMAConv

class ARMA_GNN(nn.Module):
    def __init__(self, input_dim=5, output_dim=2, edge_feat_dim=4, num_layers=8, hidden_dim=64, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim # from kwargs
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.leakyReLU_small = nn.LeakyReLU(negative_slope=0.005)
        self.num_arma_layers = num_layers

        # Pre-processing layers
        self.predense1_node = nn.Linear(self.input_dim, hidden_dim)
        self.predense2_node = nn.Linear(hidden_dim, hidden_dim)
        # self.predense1_edge = nn.Linear(edge_feat_dim, 16)
        # self.predense2_edge = nn.Linear(16, 1)

        # ARMA layers
        self.arma = ARMAConv(hidden_dim, hidden_dim, num_stacks=5, num_layers=self.num_arma_layers, shared_weights=False, act=self.leakyReLU, dropout=dropout, bias=True)

        # Post-processing layer
        self.postdense1 = nn.Linear(hidden_dim, hidden_dim)
        self.postdense2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layer
        self.readout = nn.Linear(hidden_dim, output_dim)
    
    def use_physics_loss(self):
        return False

    def is_supervised(self):
        return True

    def is_complex(self):
        return False

    def is_analytical(self):
        return False

    def forward(self, data):
        """
        # data.x: [:, p_mw, q_mvar, vm_pu, va_degree, hops_to_slack]
        # data.edge_attr: [:, trafo?, r_pu, x_pu, sc_voltage]
        """
        x = torch.nan_to_num(data.x, nan=0.0) # dim=(N, self.input_dim)
        edge_index = data.edge_index # dim=(2, 2E)
        # edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0) # dim=(2E, 4)
        
        node_emb = self.leakyReLU_small(self.predense1_node(x))
        node_emb = self.leakyReLU_small(self.predense2_node(node_emb))

        # edge_emb = self.leakyReLU(self.predense1_edge(edge_attr))
        # edge_emb = self.leakyReLU(self.predense2_edge(edge_emb))
        # edge_emb = edge_emb.reshape((-1,))


        # If edge_emb causes nan, can re-run without edge_weights.
        node_emb = self.arma(node_emb, edge_index, edge_weight=None)
        # node_emb = self.arma(node_emb, edge_index, edge_weight=edge_emb)

        node_emb = self.leakyReLU(self.postdense1(node_emb))
        node_emb = self.leakyReLU(self.postdense2(node_emb))

        return self.readout(node_emb)
