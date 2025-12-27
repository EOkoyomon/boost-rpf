import numpy as np
import torch
# from torch_sparse import SparseTensor

def extract_pandapower_matrices(ppci):
    """Extract matrices from a single pandapower network."""
    Ybus = ppci["Ybus"]
    Sbus = ppci["Sbus"]
    V = ppci["V"]
    ref = ppci["ref"]
    pv = ppci["pv"]
    pq = ppci["pq"]
    baseMVA = ppci["baseMVA"]
    
    return Ybus, Sbus, V, ref, pv, pq, baseMVA

def calculate_power_mismatch(Ybus, V, Sbus, ref, pv, pq):
    """
    Calculate power flow mismatch using pandapower's exact formulation.
    This replicates the _evaluate_Fx function for non-distributed slack case.

    Args:
        Ybus: Bus admittance matrix
        V: Complex voltage vector
        Sbus: Bus power injection vector
        ref: Reference bus indices
        pv: PV bus indices
        pq: PQ bus indices

    Returns:
        F: Power mismatch vector
        mis: Complex power mismatch at all buses
    """
    # Calculate complex power mismatch at all buses
    # This is the core power flow equation: S_calc - S_specified
    Ybus_V = torch_sparse_mm_complex(Ybus, V.unsqueeze(1)).squeeze(1)
    mis = V * torch.conj(Ybus_V) - Sbus

    # For non-distributed slack case, extract mismatches according to bus types:
    # - PV buses: only real power mismatch
    # - PQ buses: both real and reactive power mismatches
    # - Reference buses: excluded from mismatch calculation
    F = torch.cat([mis[pv].real, mis[pq].real, mis[pq].imag])

    return F, mis

def scipy_sparse_to_torch_sparse(scipy_matrix, device='cpu'):
    """Convert scipy sparse matrix to PyTorch SparseTensor."""
    scipy_matrix = scipy_matrix.tocoo()
    indices = torch.from_numpy(np.vstack((scipy_matrix.row, scipy_matrix.col))).long()
    values = torch.from_numpy(scipy_matrix.data).to(torch.complex64)
    size = scipy_matrix.shape

    return SparseTensor(row=indices[0], col=indices[1], value=values, 
                       sparse_sizes=size)

def torch_sparse_mm_complex(sparse_tensor, dense_tensor):
    """Sparse-dense matrix multiplication for complex tensors."""
    # Extract real and imaginary parts of sparse matrix values
    sparse_real = sparse_tensor.set_value(sparse_tensor.storage.value().real)
    sparse_imag = sparse_tensor.set_value(sparse_tensor.storage.value().imag)

    # Extract real and imaginary parts of dense tensor
    dense_real = dense_tensor.real
    dense_imag = dense_tensor.imag
    
    # Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    result_real = sparse_real @ dense_real - sparse_imag @ dense_imag
    result_imag = sparse_real @ dense_imag + sparse_imag @ dense_real
    
    return torch.complex(result_real, result_imag)

class OptimizedBatchPhysicsInformedLoss(torch.nn.Module):
    """
    Optimized batch-compatible physics-informed loss function.
    
    This version maintains the same computational semantics as the original
    but optimizes tensor operations and caching for better performance.
    """

    def __init__(self, device='cpu', is_complex=False):
        super(OptimizedBatchPhysicsInformedLoss, self).__init__()
        self.device = device
        self.is_complex = is_complex

    def _get_network_matrices(self, ppci):
        """Get or compute network matrices with optimized caching."""
        # Extract matrices
        Ybus, Sbus, V_ref, ref, pv, pq, baseMVA = extract_pandapower_matrices(ppci)

        # Convert to optimized PyTorch tensors
        Ybus_torch = scipy_sparse_to_torch_sparse(Ybus).to(self.device)
        Sbus_torch = torch.from_numpy(Sbus).to(torch.complex64).to(self.device)

        # Store bus type indices as tensors for faster indexing
        ref_torch = torch.from_numpy(ref).long().to(self.device)
        pv_torch = torch.from_numpy(pv).long().to(self.device)
        pq_torch = torch.from_numpy(pq).long().to(self.device)

        # Pre-compute reference solution
        V_ref_complex = torch.from_numpy(V_ref).to(torch.complex64).to(self.device)

        network_data = {
            'Ybus': Ybus_torch,
            'Sbus': Sbus_torch,
            'ref': ref_torch,
            'pv': pv_torch,
            'pq': pq_torch,
            'baseMVA': baseMVA,
            'V_ref': V_ref_complex,
        }
        return network_data
    
    def _map_net_predictions_to_ppci_optimized(self, net_predictions, network_data):
        """Optimized version of prediction mapping."""
        predictions_copy = net_predictions.clone()
        predictions_copy = predictions_copy.squeeze(1)  # Remove last dim if complex. [num_buses, 1] -> [num_buses]

        if self.is_complex:
            ppci_predictions = network_data["V_ref"].clone()
        else:
            vm_pu = torch.abs(network_data["V_ref"]).real
            va_degree = torch.angle(network_data["V_ref"]).real * (180.0 / torch.pi)
            ppci_predictions = torch.stack([vm_pu, va_degree], dim=1)


        num_predictions = len(predictions_copy)
        num_ppci_buses = len(network_data["Sbus"])

        # Simple mapping for first N buses, but skip slack bus
        min_buses = min(num_predictions, num_ppci_buses)
        ppci_predictions[1:min_buses] = predictions_copy[1:min_buses]

        return ppci_predictions
    
    def _calculate_single_network_loss_optimized(self, predictions, network_data):
        """Optimized single network loss calculation."""
        # Map predictions efficiently  
        ppci_predictions = self._map_net_predictions_to_ppci_optimized(predictions, network_data)

        if self.is_complex:
            V = ppci_predictions
        else:
            # Vectorized voltage conversion
            vm_pu = ppci_predictions[:, 0]
            va_degree = ppci_predictions[:, 1]
            va_rad = va_degree * (torch.pi / 180.0)

            # Use efficient complex number creation
            cos_va = torch.cos(va_rad)
            sin_va = torch.sin(va_rad)
            V = vm_pu * torch.complex(cos_va, sin_va)

        # Calculate power mismatch using proper bus type filtering
        F, mis = calculate_power_mismatch(network_data['Ybus'], V, network_data['Sbus'],
                                        network_data['ref'], network_data['pv'], network_data['pq'])
        return torch.sum(F**2)
    
    def forward(self, batch_predictions, batch_data):
        """
        Calculate physics-informed loss with batched matrix operations for better performance.
        Groups networks by structure and processes them in parallel.
        Args:
            batch_predictions: List of tensors with network predictions for each graph in the batch.
            batch_data: List of data objects containing network information for each graph.
        Returns:
            Average physics-informed loss across the batch.
        """
        if len(batch_predictions) == 0 or len(batch_data) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Group data by network structure (grid_name)
        groups = self._group_by_structure(batch_predictions, batch_data)
        
        batch_losses = []
        
        for group_data in groups.values():
            if len(group_data['predictions']) == 1:
                # Single graph - use original method
                predictions = group_data['predictions'][0]
                data = group_data['data'][0]
                network_data = self._get_network_matrices(data.ppci)
                loss = self._calculate_single_network_loss_optimized(predictions, network_data)
                batch_losses.append(loss)
            else:
                # Multiple graphs with same structure - use batched method
                batched_loss = self._calculate_batched_network_loss(
                    group_data['predictions'], 
                    group_data['data']
                )
                batch_losses.extend(batched_loss)
        
        # Return average loss across batch (same as original)
        return torch.stack(batch_losses).mean()

    def _group_by_structure(self, batch_predictions, batch_data):
        """Group data by network structure for batched processing."""
        groups = {}
        
        for predictions, data in zip(batch_predictions, batch_data):
            # Use grid_name and Sbus combination. Some grids can have different
            # length Sbus and vice versa.
            key = f"{data.grid_name}_{len(data.ppci['Sbus'])}"
            
            if key not in groups:
                groups[key] = {'predictions': [], 'data': []}
            
            groups[key]['predictions'].append(predictions)
            groups[key]['data'].append(data)
        
        return groups
    
    def _calculate_batched_network_loss(self, batch_predictions, batch_data):
        """Calculate loss for multiple networks with same structure in parallel."""
        if len(batch_predictions) == 0:
            return []
        
        # Get network structure from first graph (all should be identical)
        reference_data = batch_data[0]
        reference_network_data = self._get_network_matrices(reference_data.ppci)
        
        # Stack predictions for batched processing
        stacked_predictions = torch.stack(batch_predictions)  # [batch_size, num_nodes, 2] or [batch_size, num_nodes, 1] for complex
        
        # Extract and stack variable data (Sbus, V_ref) from all graphs
        sbus_batch = []
        v_ref_batch = []
        ybus_list = []
        
        for data in batch_data:
            network_data = self._get_network_matrices(data.ppci)
            sbus_batch.append(network_data['Sbus'])
            v_ref_batch.append(network_data['V_ref'])
            ybus_list.append(network_data['Ybus'])
        
        sbus_stacked = torch.stack(sbus_batch)  # [batch_size, num_buses]
        v_ref_stacked = torch.stack(v_ref_batch)  # [batch_size, num_buses]
        
        # Perform batched computation
        losses = self._calculate_batched_power_mismatch(
            stacked_predictions, 
            ybus_list,
            sbus_stacked,
            reference_network_data['ref'],
            reference_network_data['pv'],
            reference_network_data['pq'],
            v_ref_stacked
        )
        
        return losses
    
    def _calculate_batched_power_mismatch(self, batch_predictions, batch_Ybus, batch_Sbus, ref, pv, pq, batch_V_ref):
        """Calculate power mismatch for a batch of networks with identical structure."""
        batch_predictions = batch_predictions.squeeze(2)  # Remove last dim if complex. [batch_size, num_buses, 1] -> [batch_size, num_buses]
        batch_size = batch_predictions.shape[0]

        if self.is_complex:
            # Directly use complex voltages
            batch_ppci_predictions = batch_V_ref.clone()
        else:
            # Vectorized mapping of predictions to full bus vectors
            batch_vm_ref = torch.abs(batch_V_ref).real  # [batch_size, num_buses]
            batch_va_ref = torch.angle(batch_V_ref).real * (180.0 / torch.pi)  # [batch_size, num_buses]

            # Initialize with reference values
            batch_ppci_predictions = torch.stack([batch_vm_ref, batch_va_ref], dim=2)  # [batch_size, num_buses, 2]
        
        # Update with actual predictions (skip slack bus at index 0)
        num_predictions = batch_predictions.shape[1]
        num_ppci_buses = batch_ppci_predictions.shape[1]
        min_buses = min(num_predictions, num_ppci_buses)
        
        if min_buses > 1:  # Skip slack bus
            batch_ppci_predictions[:, 1:min_buses] = batch_predictions[:, 1:min_buses]
        
        if self.is_complex:
            # Directly use complex voltages
            batch_V = batch_ppci_predictions  # [batch_size, num_buses]
        else:
            # Vectorized conversion to complex voltages
            batch_vm_pu = batch_ppci_predictions[:, :, 0]  # [batch_size, num_buses]
            batch_va_degree = batch_ppci_predictions[:, :, 1]  # [batch_size, num_buses]
            batch_va_rad = batch_va_degree * (torch.pi / 180.0)

            batch_cos_va = torch.cos(batch_va_rad)
            batch_sin_va = torch.sin(batch_va_rad)
            batch_V = batch_vm_pu * torch.complex(batch_cos_va, batch_sin_va)  # [batch_size, num_buses]
        
        # Batched power mismatch calculation
        batch_losses = []
        
        # For now, still calculate each graph separately since sparse matrix ops are complex to batch
        # TODO: Could be further optimized with block-diagonal sparse matrices
        for i in range(batch_size):
            V = batch_V[i]
            Sbus = batch_Sbus[i]
            Ybus = batch_Ybus[i]
            
            # Calculate power mismatch (reuse existing function)
            F, _ = calculate_power_mismatch(Ybus, V, Sbus, ref, pv, pq)
            loss = torch.sum(F**2)
            batch_losses.append(loss)
        
        return batch_losses

    def verify_batch(self, batch_data):
        """Verify the loss function with pandapower reference solutions."""
        results = []
        pytorch_loss = 0.0
        pandapower_loss = 0.0
        
        with torch.no_grad():
            batch_predictions = [batch_data.y[(batch_data.batch == i)] for i in range(batch_data.num_graphs)]
            loss = self.forward(batch_predictions, batch_data.to_data_list())
            pytorch_loss += loss.item()*len(batch_data)

            for _, data in enumerate(batch_data.to_data_list()):
                network_data = self._get_network_matrices(data.ppci)
                
                # Convert reference solution to expected format
                vm_ref = torch.abs(network_data['V_ref']).real
                va_ref = torch.angle(network_data['V_ref']).real * (180.0 / torch.pi)
                ref_predictions = torch.stack([vm_ref, va_ref], dim=1)[:len(data.x)]
                
                # Calculate loss with reference solution
                ref_loss = self._calculate_single_network_loss_optimized(ref_predictions, network_data)
                pandapower_loss += ref_loss.item()
                
                results.append({
                    'num_buses': len(network_data['Sbus']),
                    'num_pv_buses': len(network_data['pv']),
                    'num_pq_buses': len(network_data['pq']),
                    'reference_loss': ref_loss.item()
                })

        pytorch_loss /= len(batch_data)
        pandapower_loss /= len(batch_data)
        return results, pytorch_loss, pandapower_loss

    def verify_data(self, data):
        """Verify the loss function with pandapower reference solutions."""
        results = []

        with torch.no_grad():
            network_data = self._get_network_matrices(data.ppci)

            # Convert reference solution to expected format
            vm_ref = torch.abs(network_data['V_ref']).real
            va_ref = torch.angle(network_data['V_ref']).real * (180.0 / torch.pi)
            ref_predictions = torch.stack([vm_ref, va_ref], dim=1)[:len(data.x)]

            # Calculate loss with reference solution
            ref_loss = self._calculate_single_network_loss_optimized(ref_predictions, network_data)

            results.append({
                'num_buses': len(network_data['Sbus']),
                'num_pv_buses': len(network_data['pv']),
                'num_pq_buses': len(network_data['pq']),
                'reference_loss': ref_loss.item()
            })

            print(f"Network {0}: Reference loss = {ref_loss.item():.6e}")

        return results

# Factory functions
def create_batch_physics_loss(device='cpu', is_complex=False, **kwargs):
    """Create optimized batch physics loss function."""
    return OptimizedBatchPhysicsInformedLoss(device=device, is_complex=is_complex, **kwargs)
