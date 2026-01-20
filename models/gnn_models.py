import torch
import torch.nn as nn

# from complexPyTorch.complexFunctions import complex_relu
# from complexPyTorch.complexLayers import ComplexLinear
from torch_geometric.nn import MessagePassing


class GNNLayer(MessagePassing):
    def __init__(self, in_c, out_c):
        super().__init__(aggr="add")
        # Linear applied after aggregation
        self.lin = nn.Linear(in_c, out_c)
        # Linear for the root/self contribution
        self.lin_root = nn.Linear(in_c, out_c)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * self.lin(x_j)

    def update(self, inputs, x):
        return self.lin_root(x) + inputs


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
            self.layers.append(GNNLayer(self.hidden_dim, self.hidden_dim))
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
        #   x features: [p_mw, q_mvar, hops_to_slack]
        #   edge_attr features: [r_pu, x_pu]
        #   y labels: [vm_pu, va_degree]
        #   slack_info (global): [slack_vm_pu, slack_va_degree, slack_r_pu, slack_x_pu]

        x = torch.nan_to_num(data.x, nan=0.0)
        edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0)
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


class NormedGNN_Residuals(NormedGNN):
    def __init__(self, input_dim=7, output_dim=2, num_layers=7):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            residuals=True,
            physics_loss=False,
            supervised=True,
            complex=False,
        )

class NormedGNN_PhysicsLoss_Supervised(NormedGNN):
    def __init__(self, input_dim=7, output_dim=2, num_layers=7):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            residuals=False,
            physics_loss=True,
            supervised=True,
            complex=False,
        )


class NormedGNN_Complex(nn.Module):
    class ComplexBatchNorm1d(nn.Module):
        def __init__(self, num_features):
            super().__init__()
            self.bn_real = nn.BatchNorm1d(num_features)
            self.bn_imag = nn.BatchNorm1d(num_features)

        def forward(self, x):
            return torch.complex(self.bn_real(x.real), self.bn_imag(x.imag))

    class ComplexGNNLayer(MessagePassing):
        def __init__(self, in_c, out_c):
            super().__init__(aggr="add")
            # Linear applied after aggregation
            self.lin = ComplexLinear(in_c, out_c)
            # Linear for the root/self contribution
            self.lin_root = ComplexLinear(in_c, out_c)

        def forward(self, x, edge_index, edge_weight=None):
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
            return self.propagate(edge_index, x=x, edge_weight=edge_weight)

        def message(self, x_j, edge_weight):
            return edge_weight.view(-1, 1).to(x_j.dtype) * self.lin(x_j)

        def update(self, inputs, x):
            return self.lin_root(x) + inputs

    class ComplexGraphConv(MessagePassing):
        def __init__(self, in_c, out_c):
            super().__init__(aggr="add")
            # Linear applied after aggregation (like torch_geometric GraphConv)
            self.lin = ComplexLinear(in_c, out_c)
            # Linear for the root/self contribution
            self.lin_root = ComplexLinear(in_c, out_c)

        def forward(self, x, edge_index, edge_weight=None):
            # Allow x to be a single Tensor or a pair (x_src, x_dst)
            if isinstance(x, torch.Tensor):
                x = (x, x)

            out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

            # apply linear after aggregation
            out = self.lin(out)

            x_r = x[1]
            if x_r is not None:
                out = out + self.lin_root(x_r)

            return out

        def message(self, x_j, edge_weight=None):
            if edge_weight is None:
                return x_j
            return edge_weight.to(x_j.dtype).view(-1, 1) * x_j

    def __init__(
        self,
        input_dim=4,
        output_dim=1,
        num_layers=7,
        residuals=False,
        physics_loss=False,
        supervised=True,
    ):
        super().__init__()

        self.residuals = residuals
        self.physics_loss = physics_loss
        self.supervised = supervised
        self.complex = True
        self.hidden_dim = 128
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.leakyReLU = complex_relu

        # Node feature processing layers
        self.predense1 = ComplexLinear(self.input_dim, self.hidden_dim * 2)
        self.prenorm = NormedGNN_Complex.ComplexBatchNorm1d(self.hidden_dim * 2)

        self.predense2 = ComplexLinear(self.hidden_dim * 2, self.hidden_dim)
        self.prenorm2 = NormedGNN_Complex.ComplexBatchNorm1d(self.hidden_dim)

        # GNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            self.layers.append(
                NormedGNN_Complex.ComplexGNNLayer(self.hidden_dim, self.hidden_dim)
            )
            self.norms.append(NormedGNN_Complex.ComplexBatchNorm1d(self.hidden_dim))

        self.postdense1 = ComplexLinear(self.hidden_dim, self.hidden_dim * 2)
        self.postnorm = NormedGNN_Complex.ComplexBatchNorm1d(self.hidden_dim * 2)

        self.readout = ComplexLinear(self.hidden_dim * 2, self.output_dim)

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
        #   x features: [complex_power, hops_to_slack]
        #   edge_attr features: [complex_impedance]
        #   y labels: [complex_voltage]
        #   slack_info (global): [slack_complex_voltage, slack_complex_impedance]

        x = torch.nan_to_num(data.x, nan=0.0)
        edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0)
        edge_index = data.edge_index

        # If 'batch' does not exist, make batch of 1
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = data.x.new_zeros(x.size(0), dtype=torch.long)

        # Handle batched data: slack_info contains slack info for each graph in the batch
        # data.batch maps each node to its corresponding graph in the batch
        # data.slack_info has shape [batch_size * 2] - need to reshape and index properly

        batch_size = data.batch.max().item() + 1  # Number of graphs in batch
        slack_info_per_graph = data.slack_info.view(
            batch_size, 2
        )  # Reshape to [batch_size, 2]

        # For each node, get the slack info from its corresponding graph
        # data.batch[i] tells us which graph node i belongs to
        node_slack_info = slack_info_per_graph[data.batch]  # Shape: [num_nodes, 2]

        # Append slack info to each node feature
        # Now x has shape: [complex_power, hops_to_slack, slack_complex_voltage, slack_complex_impedance] (4 features)
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
            x = x + orig_x[:, 2:3]  # Add slack_vm_pu and slack_va_degree (as one complex voltage)
        return x

