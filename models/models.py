import numpy as np
import torch
import torch.nn as nn
from complexPyTorch.complexFunctions import complex_relu
from complexPyTorch.complexLayers import ComplexLinear
from torch_geometric.nn import GATv2Conv, GCNConv, GraphConv, MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from models.lindistflow import calculate_distflow_iterative, calculate_lindistflow_iterative


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

class NormedGNN_Residuals_PhysicsLoss(NormedGNN):
    def __init__(self, input_dim=7, output_dim=2, num_layers=7):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            residuals=True,
            physics_loss=True,
            supervised=True,
            complex=False,
        )


class NormedGNN_PhysicsLoss(NormedGNN):
    def __init__(self, input_dim=7, output_dim=2, num_layers=7):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            residuals=False,
            physics_loss=True,
            supervised=False,
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

class NormedGNN_Complex_PhysicsLoss(NormedGNN_Complex):
    def __init__(self, input_dim=4, output_dim=1, num_layers=7):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            residuals=False,
            physics_loss=True,
            supervised=True,
        )

class NormedGNN_Complex_Residuals(NormedGNN_Complex):
    def __init__(self, input_dim=4, output_dim=1, num_layers=7):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            residuals=True,
            physics_loss=False,
            supervised=True,
        )

# GCN class from ENGAGE, adapted for new data format.
class GCN_ENGAGE(nn.Module):
    def __init__(self, input_dim=7, num_layers=8):
        super().__init__()
        # Constants
        self.input_dim = input_dim
        self.num_gcn_layers = num_layers
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.leakyReLU_small = nn.LeakyReLU(negative_slope=0.005)

        # Pre-processing layers
        self.predense1_node = nn.Linear(self.input_dim, 64)
        self.predense2_node = nn.Linear(64, 64)
        self.predense1_edge = nn.Linear(2, 16)
        self.predense2_edge = nn.Linear(16, 1)

        self.gcn_layers = nn.ModuleList(
            [GCNConv(64, 64, normalize=True) for i in range(self.num_gcn_layers)]
        )

        # Post-processing layer
        self.postdense1 = nn.Linear(64 + self.input_dim, 64)
        self.postdense2 = nn.Linear(64, 64)

        # Output layer
        self.readout = nn.Linear(64, 2)

    def use_physics_loss(self):
        return False

    def is_supervised(self):
        return True

    def is_complex(self):
        return False

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
        node_emb = self.leakyReLU(self.predense1_node(x))
        node_emb = self.leakyReLU(self.predense2_node(node_emb))
        edge_emb = self.leakyReLU(self.predense1_edge(edge_attr))
        # Using a leaky relu with too large of negative_slope can lead to
        # sqrt of a negative number in GCNConv, so we use different leakyReLU
        # in last step.
        edge_emb = self.leakyReLU_small(self.predense2_edge(edge_emb))
        edge_emb = edge_emb.reshape((-1,))

        # GNN Layers
        for _, layer in enumerate(self.gcn_layers):
            node_emb = self.leakyReLU(
                layer(x=node_emb, edge_index=edge_index, edge_weight=edge_emb)
            )

        # Post-processing
        node_emb = torch.cat([orig_x, node_emb], 1)
        node_emb = self.leakyReLU(self.postdense1(node_emb))
        node_emb = self.leakyReLU(self.postdense2(node_emb))

        pred = self.readout(node_emb)

        return pred

class NormedGAT(NormedGNN):
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
        super().__init__(
            input_dim,
            output_dim,
            num_layers,
            residuals,
            physics_loss,
            supervised,
            complex,
        )
        # Replace GNN layers with GATv2Conv layers
        self.layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.layers.append(GATv2Conv(self.hidden_dim, self.hidden_dim // 4, heads=4))

class NormedGAT_Wide(NormedGNN):
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
        super().__init__(
            input_dim,
            output_dim,
            num_layers,
            residuals,
            physics_loss,
            supervised,
            complex,
        )
        # Replace GNN layers with GATv2Conv layers
        self.layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.layers.append(GATv2Conv(self.hidden_dim, self.hidden_dim, heads=4, concat=False))

class NormedGAT_Residuals(NormedGAT):
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

class NormedGAT_Wide_Residuals(NormedGAT_Wide):
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

class NormedGAT_PhysicsLoss_Supervised(NormedGAT):
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

class NormedGAT_Wide_PhysicsLoss_Supervised(NormedGAT_Wide):
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

class NormedGAT_Complex(NormedGNN_Complex):
    class ComplexGATLayer(MessagePassing):
        """Complex GATv2 layer adapted from torch_geometric GATv2Conv."""
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            **kwargs,
        ):
            super().__init__(node_dim=0, aggr="add", **kwargs)

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.heads = heads
            self.concat = concat

            # Complex linear transformations for source and target nodes
            self.lin_l = ComplexLinear(in_channels, heads * out_channels)
            self.lin_r = ComplexLinear(in_channels, heads * out_channels)

            # Complex attention parameters (separate real and imaginary parts)
            self.att_real = nn.Parameter(torch.empty(1, heads, out_channels))
            self.att_imag = nn.Parameter(torch.empty(1, heads, out_channels))

            self.reset_parameters()

        def reset_parameters(self) -> None:
            """Initialize parameters using Glorot, like in GATv2Conv."""
            super().reset_parameters()

            # Initialize linear layers
            glorot(self.lin_l.fc_r.weight)
            glorot(self.lin_l.fc_i.weight)
            glorot(self.lin_r.fc_r.weight)
            glorot(self.lin_r.fc_i.weight)

            # Initialize biases
            if self.lin_l.fc_r.bias is not None:
                zeros(self.lin_l.fc_r.bias)
                zeros(self.lin_l.fc_i.bias)
            if self.lin_r.fc_r.bias is not None:
                zeros(self.lin_r.fc_r.bias)
                zeros(self.lin_r.fc_i.bias)

            # Initialize attention parameters
            glorot(self.att_real)
            glorot(self.att_imag)

        def forward(self, x, edge_index, return_attention_weights = False):
            """
            Forward pass of the Complex GATv2 layer.

            Returns:
                torch.Tensor: Complex output node features of shape [N, out_channels*heads] if concat=True,
                    or [N, out_channels] if concat=False, with complex dtype.
            """
            H, C = self.heads, self.out_channels

            assert x.dim() == 2
            assert x.size(1) == self.in_channels, \
                f"Expected input with {self.in_channels} features, got {x.size(1)}"
            assert x.dtype in [torch.complex64, torch.complex128], \
                f"Expected complex tensor, got {x.dtype}"

            x_l = self.lin_l(x).view(-1, H, C)
            x_r = self.lin_r(x).view(-1, H, C)

            # Add self-loops
            num_nodes = x_l.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

            # Compute attention coefficients. Edge updates done in edge_update func below.
            # edge_updater_type: (x: PairTensor, edge_attr: OptTensor)
            alpha = self.edge_updater(edge_index, x=(x_l, x_r))

            # Propagate messages. Done in message funcs below. Default 'add' aggr.
            # propagate_type: (x: PairTensor, alpha: Tensor)
            out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

            # Concatenate or average heads
            if self.concat:
                out = out.view(-1, self.heads * self.out_channels)
            else:
                out = out.mean(dim=1)

            if return_attention_weights:
                return out, (edge_index, alpha)
            else:
                return out

        def edge_update(
            self,
            x_i: torch.Tensor,
            x_j: torch.Tensor,
            index: torch.Tensor,
            ptr: OptTensor,
            dim_size: int,
        ) -> torch.Tensor:
            """Compute attention coefficients for each edge."""
            # x_i and x_j are complex tensors
            # When we pass x=(x_l, x_r) to edge_updater:
            # - x_i comes from x_r (target node transformation)
            # - x_j comes from x_l (source node transformation)
            # They have already been multiplied by the sender and receiver matrices.
            # In GATv2, attention is: a^T * LeakyReLU(W_s * x_i + W_t * x_j) ... i and j apparently flipped, but we sum them so doesnt matter.

            # Complex addition
            x = x_i + x_j

            # Complex relu
            x = complex_relu(x)

            # In GATv2 we need to do a dot product with an attention vector, which is complex in this case.
            # Complex dot product with attention vector.
            # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
            # For the dot product, we sum over the last dimension
            x_real, x_imag = x.real, x.imag
            alpha_real = (x_real * self.att_real - x_imag * self.att_imag).sum(dim=-1)
            alpha_imag = (x_real * self.att_imag + x_imag * self.att_real).sum(dim=-1)

            # Use magnitude of complex number for attention score
            alpha = torch.sqrt(alpha_real ** 2 + alpha_imag ** 2)

            # Apply softmax normalization
            alpha = softmax(alpha, index, ptr, dim_size)

            return alpha

        def message(self, x_j: torch.Tensor, alpha: torch.Tensor):
            """Compute messages to be aggregated."""
            # x_j is a complex tensor

            # Apply attention weights (alpha is real-valued), by scaling both real and imaginary part.
            # Geometrically, this scales the vector in the complex plane, without changing its direction. 
            out = x_j * alpha.unsqueeze(-1)

            return out

    def __init__(
        self,
        input_dim=4,
        output_dim=1,
        num_layers=7,
        residuals=False,
        physics_loss=False,
        supervised=True,
    ):
        super().__init__(
            input_dim,
            output_dim,
            num_layers,
            residuals,
            physics_loss,
            supervised,
        )
        # Replace GNN layers with GATv2Conv layers
        self.layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.layers.append(NormedGAT_Complex.ComplexGATLayer(self.hidden_dim, self.hidden_dim // 4, heads=4))

class NormedGAT_Wide_Complex(NormedGNN_Complex):
    def __init__(
        self,
        input_dim=4,
        output_dim=1,
        num_layers=7,
        residuals=False,
        physics_loss=False,
        supervised=True,
    ):
        super().__init__(
            input_dim,
            output_dim,
            num_layers,
            residuals,
            physics_loss,
            supervised,
        )
        # Replace GNN layers with GATv2Conv layers
        self.layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.layers.append(NormedGAT_Complex.ComplexGATLayer(self.hidden_dim, self.hidden_dim, heads=4, concat=False))

class DC_PF(nn.Module):
    """ Implements the DC Power Flow as a neural network module."""
    def __init__(self):
        super().__init__()

    def use_physics_loss(self):
        return False

    def is_supervised(self):
        return False

    def is_complex(self):
        return False

    def is_analytical(self):
        return True

    def forward(self, data):
        return data.dc_pf

class DC_PF_Slack(DC_PF):
    """ Sets all voltage magnitudes to slack bus voltage magnitude."""
    def __init__(self):
        super().__init__()

    def forward(self, data):
        out = data.dc_pf
        out[:, 0] = torch.ones(len(out))*data.slack_info[0]
        return out

class LinDistFlow(nn.Module):
    def __init__(self):
        super().__init__()

    def use_physics_loss(self):
        return False

    def is_supervised(self):
        return False

    def is_complex(self):
        return False

    def is_analytical(self):
        return True

    def forward(self, data):
        vm_predictions, va_predictions = calculate_lindistflow_iterative(data, slack_index=0, slack_vm_pu=data.slack_info[0])
        out = torch.stack([torch.tensor(vm_predictions), torch.tensor(va_predictions)], dim=1)
        return out

class DistFlow(nn.Module):
    def __init__(self):
        super().__init__()

    def use_physics_loss(self):
        return False

    def is_supervised(self):
        return False

    def is_complex(self):
        return False

    def is_analytical(self):
        return True

    def forward(self, data):
        vm_predictions, va_predictions = calculate_distflow_iterative(data, slack_index=0, slack_vm_pu=data.slack_info[0])
        out = torch.stack([torch.tensor(vm_predictions), torch.tensor(va_predictions)], dim=1)
        return out
    