# import numpy as np
# import torch
# import torch.nn as nn


# class PathTransformerModel(nn.Module):
#     """
#     Transformer model for sequence-to-sequence voltage prediction along grid paths.
    
#     Treats the entire path as a sequence (like a sentence):
#     - Input: Covariates for each node in the path [r_ij, x_ij, P_j, Q_j, P_agg_j, Q_agg_j, V_LDF_j, theta_LDF_j]
#     - Output: Predicted voltages [V_j, theta_j] for each node in the path
#     - The slack voltage is provided as context (prepended or used as encoder input)
    
#     Shorter paths are padded to the maximum path length in the batch.
#     """
    
#     def __init__(self, input_dim=8, output_dim=2, hidden_dim=64, num_heads=4, 
#                  num_encoder_layers=3, num_decoder_layers=3, dropout=0.1, max_seq_len=100):
#         super().__init__()
        
#         self.input_dim = input_dim  # Covariate features
#         self.output_dim = output_dim  # [V, theta]
#         self.hidden_dim = hidden_dim
#         self.max_seq_len = max_seq_len
        
#         # Input embedding for covariates
#         self.input_embed = nn.Linear(input_dim, hidden_dim)
        
#         # Embedding for target (voltage) - used in decoder
#         self.target_embed = nn.Linear(output_dim, hidden_dim)
        
#         # Positional encoding
#         self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
#         nn.init.normal_(self.pos_encoding, std=0.02)
        
#         # Transformer
#         self.transformer = nn.Transformer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dim_feedforward=hidden_dim * 4,
#             dropout=dropout,
#             batch_first=True
#         )
        
#         # Output projection
#         self.output_proj = nn.Linear(hidden_dim, output_dim)
        
#     def forward(self, covariates, slack_voltage, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
#         """
#         Forward pass.
        
#         Args:
#             covariates: [batch, seq_len, input_dim] - path covariates
#             slack_voltage: [batch, 1, output_dim] - slack bus voltage as start token
#             src_mask: Optional source mask
#             tgt_mask: Optional target mask (causal)
#             src_padding_mask: [batch, seq_len] - True for padded positions
#             tgt_padding_mask: [batch, seq_len] - True for padded positions
            
#         Returns:
#             predictions: [batch, seq_len, output_dim] - predicted voltages for each position
#         """
#         batch_size, seq_len, _ = covariates.shape
        
#         # Embed covariates and add positional encoding
#         src = self.input_embed(covariates) + self.pos_encoding[:, :seq_len, :]
        
#         # Create target sequence: start with slack voltage, then zeros for positions to predict
#         # We'll use teacher forcing during training (shift targets)
#         tgt_input = torch.zeros(batch_size, seq_len, self.output_dim, device=covariates.device)
#         tgt_input[:, 0, :] = slack_voltage.squeeze(1)  # First position is slack voltage
        
#         # Embed target and add positional encoding
#         tgt = self.target_embed(tgt_input) + self.pos_encoding[:, :seq_len, :]
        
#         # Generate causal mask for decoder (can't look ahead)
#         if tgt_mask is None:
#             tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=covariates.device)
        
#         # Transformer forward
#         output = self.transformer(
#             src, tgt,
#             src_mask=src_mask,
#             tgt_mask=tgt_mask,
#             src_key_padding_mask=src_padding_mask,
#             tgt_key_padding_mask=tgt_padding_mask
#         )
        
#         # Project to output dimension
#         predictions = self.output_proj(output)
        
#         return predictions


# class PathTransformerWrapper:
#     """
#     Wrapper for PathTransformerModel for sequential voltage prediction along grid paths.
    
#     This model processes entire paths as sequences (like sentences in NLP):
#     - Input: Full path covariates [r_ij, x_ij, P_j, Q_j, P_agg_j, Q_agg_j, V_LDF_j, theta_LDF_j]
#     - Output: All voltages along the path [V_j, theta_j] predicted in one forward pass
#     - Padding: Shorter paths are padded to batch max length
    
#     The slack voltage is used as the "start token" - the only known voltage.
#     """
    
#     def __init__(self, hidden_dim=64, num_heads=4, num_encoder_layers=3, num_decoder_layers=3,
#                  dropout=0.1, max_seq_len=100, n_epochs=50, batch_size=64, lr=1e-3, random_state=42):
#         """
#         Initialize the PathTransformer wrapper.
#         """
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.num_encoder_layers = num_encoder_layers
#         self.num_decoder_layers = num_decoder_layers
#         self.dropout = dropout
#         self.max_seq_len = max_seq_len
#         self.n_epochs = n_epochs
#         self.batch_size = batch_size
#         self.lr = lr
#         self.random_state = random_state
        
#         torch.manual_seed(random_state)
#         np.random.seed(random_state)
        
#         self.model = PathTransformerModel(
#             input_dim=8,  # covariates
#             output_dim=2,  # [V, theta]
#             hidden_dim=hidden_dim,
#             num_heads=num_heads,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dropout=dropout,
#             max_seq_len=max_seq_len
#         )
        
#         self._is_fitted = False
#         self.device = torch.device('cpu')  # Use CPU to avoid MPS issues
#         self.model.to(self.device)
        
#     @property
#     def __name__(self):
#         return "PathTransformerWrapper"
    
#     def _prepare_batch(self, target_series_list, covariate_series_list):
#         """
#         Prepare a padded batch from lists of TimeSeries.
        
#         Returns:
#             covariates: [batch, max_len, 8] padded covariate tensor
#             targets: [batch, max_len, 2] padded target tensor  
#             slack_voltages: [batch, 1, 2] slack voltage for each path
#             padding_mask: [batch, max_len] True for padded positions
#             lengths: [batch] original lengths
#         """
#         batch_size = len(target_series_list)
        
#         # Get lengths and find max
#         lengths = [len(ts) for ts in target_series_list]
#         max_len = max(lengths)
        
#         # Initialize padded tensors
#         covariates = torch.zeros(batch_size, max_len, 8, dtype=torch.float32)
#         targets = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)
#         slack_voltages = torch.zeros(batch_size, 1, 2, dtype=torch.float32)
#         padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)  # True = masked
        
#         for i, (target_ts, cov_ts) in enumerate(zip(target_series_list, covariate_series_list)):
#             seq_len = lengths[i]
            
#             # Extract values
#             target_vals = target_ts.values().astype(np.float32)
#             cov_vals = cov_ts.values().astype(np.float32)
            
#             # Fill tensors
#             covariates[i, :seq_len, :] = torch.from_numpy(cov_vals)
#             targets[i, :seq_len, :] = torch.from_numpy(target_vals)
#             slack_voltages[i, 0, :] = torch.from_numpy(target_vals[0])  # First position is slack
#             padding_mask[i, :seq_len] = False  # Not masked for real positions
            
#         return covariates, targets, slack_voltages, padding_mask, lengths

#     def get_validation_error(self):
#         """Get the final validation error after training."""
#         if not self._is_fitted:
#             raise RuntimeError("Model must be fitted before getting validation error.")
#         return self._val_error
    
#     def fit(self, target_series_train, covariate_series_train, target_series_val, covariate_series_val, verbose=True):
#         """
#         Fit the model on a list of path sequences.
#         """
#         self.model.train()
#         optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
#         criterion = nn.MSELoss(reduction='none')  # Per-element loss for masking
        
#         # Filter sequences with length >= 2 (need at least slack + one node)
#         train_targets = []
#         train_covs = []
#         for t, c in zip(target_series_train, covariate_series_train):
#             if len(t) >= 2:
#                 train_targets.append(t)
#                 train_covs.append(c)
        
#         val_targets = []
#         val_covs = []
#         for t, c in zip(target_series_val, covariate_series_val):
#             if len(t) >= 2:
#                 val_targets.append(t)
#                 val_covs.append(c)
        
#         if verbose:
#             print(f"Training on {len(train_targets)} sequences, validating on {len(val_targets)}")
        
#         n_batches = (len(train_targets) + self.batch_size - 1) // self.batch_size
        
#         for epoch in range(self.n_epochs):
#             print(f"Epoch {epoch+1}/{self.n_epochs}")
#             # Shuffle training data
#             indices = np.random.permutation(len(train_targets))
#             train_targets_shuffled = [train_targets[i] for i in indices]
#             train_covs_shuffled = [train_covs[i] for i in indices]
            
#             epoch_loss = 0.0
#             for batch_idx in tqdm(range(n_batches)):
#                 start_idx = batch_idx * self.batch_size
#                 end_idx = min(start_idx + self.batch_size, len(train_targets))
                
#                 batch_targets = train_targets_shuffled[start_idx:end_idx]
#                 batch_covs = train_covs_shuffled[start_idx:end_idx]
                
#                 covariates, targets, slack_voltages, padding_mask, lengths = self._prepare_batch(
#                     batch_targets, batch_covs
#                 )
                
#                 covariates = covariates.to(self.device)
#                 targets = targets.to(self.device)
#                 slack_voltages = slack_voltages.to(self.device)
#                 padding_mask = padding_mask.to(self.device)
                
#                 optimizer.zero_grad()
                
#                 # Forward pass
#                 predictions = self.model(
#                     covariates, slack_voltages,
#                     src_padding_mask=padding_mask,
#                     tgt_padding_mask=padding_mask
#                 )
                
#                 # Compute masked loss (ignore padding and slack position)
#                 loss_mask = ~padding_mask  # [batch, seq_len]
#                 loss_mask[:, 0] = False  # Don't compute loss on slack voltage (it's given)
                
#                 loss = criterion(predictions, targets)  # [batch, seq_len, 2]
#                 loss = loss.mean(dim=-1)  # [batch, seq_len]
#                 loss = (loss * loss_mask).sum() / loss_mask.sum()
                
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#                 optimizer.step()
                
#                 epoch_loss += loss.item()
            
#             epoch_loss /= n_batches
            
#             if verbose and (epoch + 1) % 10 == 0:
#                 print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.6f}")
        
#         # Compute validation error
#         self.model.eval()
#         with torch.no_grad():
#             covariates, targets, slack_voltages, padding_mask, lengths = self._prepare_batch(
#                 val_targets, val_covs
#             )
#             covariates = covariates.to(self.device)
#             targets = targets.to(self.device)
#             slack_voltages = slack_voltages.to(self.device)
#             padding_mask = padding_mask.to(self.device)
            
#             predictions = self.model(covariates, slack_voltages, src_padding_mask=padding_mask, tgt_padding_mask=padding_mask)
            
#             loss_mask = ~padding_mask
#             loss_mask[:, 0] = False
            
#             val_loss = criterion(predictions, targets).mean(dim=-1)
#             val_loss = (val_loss * loss_mask).sum() / loss_mask.sum()
#             self._val_error = val_loss.item()
        
#         if verbose:
#             print(f"Validation MSE: {self._val_error:.6f}")
        
#         self._is_fitted = True
    
#     def predict(self, sample):
#         """
#         Predict voltages for all nodes in a sample.
        
#         For each path, predicts the entire sequence in one forward pass,
#         then aggregates predictions for nodes that appear in multiple paths.
#         """
#         self.model.eval()
#         num_nodes = sample['num_nodes']
#         paths = sample['paths']
        
#         predictions = {i: [] for i in range(num_nodes)}
        
#         # Get slack voltage from first path
#         slack_voltage = paths[0]['target_series'][0].values().flatten()
#         predictions[0] = [slack_voltage]
        
#         with torch.no_grad():
#             for path_info in paths:
#                 path = path_info['path']
#                 if len(path) <= 1:
#                     continue
                
#                 # Prepare single sequence
#                 target_ts = path_info['target_series']
#                 cov_ts = path_info['covariate_series']
                
#                 covariates, targets, slack_voltages, padding_mask, lengths = self._prepare_batch(
#                     [target_ts], [cov_ts]
#                 )
                
#                 covariates = covariates.to(self.device)
#                 slack_voltages = slack_voltages.to(self.device)
#                 padding_mask = padding_mask.to(self.device)
                
#                 # Predict
#                 preds = self.model(covariates, slack_voltages, src_padding_mask=padding_mask, tgt_padding_mask=padding_mask)
#                 preds = preds[0].cpu().numpy()  # [seq_len, 2]
                
#                 # Store predictions for each node in path (skip slack at index 0)
#                 for i, node in enumerate(path[1:], start=1):
#                     predictions[node].append(preds[i])
        
#         # Average predictions for nodes appearing in multiple paths
#         result = np.zeros((num_nodes, 2))
#         for node in range(num_nodes):
#             if predictions[node]:
#                 result[node] = np.mean(predictions[node], axis=0)
        
#         return result

#     def is_fitted(self):
#         return self._is_fitted

#     def use_physics_loss(self):
#         return False

#     def is_supervised(self):
#         return True

#     def is_complex(self):
#         return False

#     def is_analytical(self):
#         return False

#     def forward(self, data):
#         raise NotImplementedError("Use the predict() method for sequential prediction.")


# class TransformerModel_Linear(PathTransformerWrapper):
#     """Path Transformer model for voltage prediction."""
#     def __init__(self, random_state=42):
#         super().__init__(
#             hidden_dim=64,
#             num_heads=4,
#             num_encoder_layers=2,
#             num_decoder_layers=2,
#             dropout=0.1,
#             max_seq_len=150,
#             n_epochs=2,
#             batch_size=2048,
#             lr=1e-3,
#             random_state=random_state
#         )
