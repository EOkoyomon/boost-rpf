import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv, MessagePassing


class NormedGNN(nn.Module):
    def __init__(
        self,
        input_dim=7,
        output_dim=2,
        num_layers=7,
        residuals=False,
        physics_loss=False,
        supervised=True,
        complex=False,
    ):
        super().__init__()

        self.residuals = residuals
        self.physics_loss = physics_loss
        self.supervised = supervised
        self.complex = complex
        self.hidden_dim = 128
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.leakyReLU = nn.functional.relu

        # Node feature processing layers
        self.predense1 = nn.Linear(self.input_dim, self.hidden_dim * 2)
        self.prenorm = nn.BatchNorm1d(self.hidden_dim * 2)

        self.predense2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.prenorm2 = nn.BatchNorm1d(self.hidden_dim)

        # GNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            self.layers.append(GraphConv(self.hidden_dim, self.hidden_dim))
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))

        # Post-processing layers
        self.postdense1 = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.postnorm = nn.BatchNorm1d(self.hidden_dim * 2)

        self.readout = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def use_physics_loss(self):
        return self.physics_loss

    def is_supervised(self):
        return self.supervised

    def is_complex(self):
        return self.complex

    def is_analytical(self):
        return False

    def forward(self, data):
        # Data format:
        #   x features: [p_mw, q_mvar, vm_pu, va_degree, hops_to_slack]
        #   edge_attr features: [trafo?, r_pu, x_pu, sc_voltage]
        #   y labels: [vm_pu, va_degree]
        #   slack_info (global): [slack_vm_pu, slack_va_degree, slack_r_pu, slack_x_pu]

        x = torch.nan_to_num(data.x, nan=0.0)[:, [0, 1, 4]]  # Keep only [p_mw, q_mvar, hops_to_slack]
        edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0) # Unused in this model
        edge_index = data.edge_index

        # If 'batch' does not exist, make batch of 1
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = data.x.new_zeros(x.size(0), dtype=torch.long)

        # Handle batched data: slack_info contains slack info for each graph in the batch
        # data.batch maps each node to its corresponding graph in the batch
        # data.slack_info has shape [batch_size * 4] - need to reshape and index properly

        batch_size = data.batch.max().item() + 1  # Number of graphs in batch
        slack_info_per_graph = data.slack_info.view(
            batch_size, 4
        )  # Reshape to [batch_size, 4]

        # For each node, get the slack info from its corresponding graph
        # data.batch[i] tells us which graph node i belongs to
        node_slack_info = slack_info_per_graph[data.batch]  # Shape: [num_nodes, 4]

        # Append slack info to each node feature
        # Now x has shape: [p_mw, q_mvar, hops_to_slack, slack_vm_pu, slack_va_degree, slack_r_pu, slack_x_pu] (7 features)
        x = torch.cat([x, node_slack_info], dim=1)

        orig_x = x

        # Pre-processing
        x = self.leakyReLU(self.prenorm(self.predense1(x)))
        x = self.leakyReLU(self.prenorm2(self.predense2(x)))

        # GNN Layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = self.norms[i](x)
            x = self.leakyReLU(x)

        # Post-processing
        x = self.postdense1(x)
        x = self.postnorm(x)
        x = self.leakyReLU(x)

        # Readout
        x = self.readout(x)
        if self.residuals:
            # Adding the slack bus's voltage components makes the NN's task to predict the residuals,
            x = x + orig_x[:, 3:5]  # Add slack_vm_pu and slack_va_degree
        return x
