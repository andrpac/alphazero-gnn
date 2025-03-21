"""
Core GNN implementation for enhancing AlphaZero with path-based processing.
These components are used by game-specific GNN wrapper classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    """
    Graph Neural Network layer that processes a collection of state representations
    from the visited path in MCTS. Uses attention mechanism to weight the importance
    of different states.
    """
    def __init__(self, feature_dim):
        super(GNNLayer, self).__init__()
        self.feature_dim = feature_dim
        
        # Position-aware processing
        self.max_depth = 32
        self.position_embedding = nn.Embedding(self.max_depth, feature_dim)
        
        # Attention mechanism to weight states based on relevance to current state
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Feature update network
        self.update_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Gate for controlled information flow
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
    
    def compute_attention(self, target_feat, source_feat):
        """Compute attention score between target and source features"""
        combined = torch.cat([target_feat, source_feat], dim=1)
        return torch.sigmoid(self.attention(combined))
    
    def forward(self, features, depths=None):
        """
        Process features through one GNN layer, updating the target state (first in list)
        based on information from all other states in the path.
        
        Args:
            features: Tensor of shape [num_states, feature_dim]
                     The first state (index 0) is the target state to update
            depths: Optional tensor of shape [num_states] indicating the depth
                   of each state in the MCTS path
            
        Returns:
            Updated features tensor of the same shape
        """
        if features.size(0) <= 1:
            # Nothing to propagate if only one state or no states
            return features
            
        # Apply position embeddings if depths are provided
        if depths is not None:
            # Ensure depths are within the supported range
            depths = torch.clamp(depths, 0, self.max_depth - 1)
            pos_emb = self.position_embedding(depths)
            features = features + pos_emb
            
        # The first feature vector is the target state
        target_feat = features[0:1]  # Keep dimension: [1, feature_dim]
        
        # Process messages from all other states (the rest of the path)
        path_states = features[1:]  # Shape: [path_length-1, feature_dim]
        
        if path_states.size(0) == 0:
            return features
        
        # Compute attention weights for each path state
        attention_weights = []
        for i in range(path_states.size(0)):
            source_feat = path_states[i:i+1]  # Keep dimension: [1, feature_dim]
            attn = self.compute_attention(target_feat, source_feat)
            attention_weights.append(attn)
        
        # Stack attention weights
        attention_weights = torch.cat(attention_weights, dim=0)  # [path_length-1]
        
        # Normalize attention weights
        if attention_weights.sum() > 0:
            attention_weights = attention_weights / attention_weights.sum()
        
        # Apply attention weights to path states
        weighted_path_feats = path_states * attention_weights.view(-1, 1)
        
        # Aggregate weighted features
        aggregated_path_info = weighted_path_feats.sum(dim=0, keepdim=True)  # [1, feature_dim]
        
        # Update target state with aggregated path information using gating
        combined = torch.cat([target_feat, aggregated_path_info], dim=1)  # [1, feature_dim*2]
        gate_value = self.gate(combined)
        update_value = self.update_net(combined)
        updated_target = target_feat + gate_value * update_value  # [1, feature_dim]
        
        # Return updated target state along with original path states
        return torch.cat([updated_target, path_states], dim=0)

class GNNProcessor(nn.Module):
    """
    GNN processor that enhances the target state representation using 
    information from states visited in the current MCTS path.
    """
    def __init__(self, feature_dim, num_layers=2):
        super(GNNProcessor, self).__init__()
        self.layers = nn.ModuleList([GNNLayer(feature_dim) for _ in range(num_layers)])
    
    def forward(self, features, depths=None):
        """
        Process features through multiple GNN layers
        
        Args:
            features: Tensor of shape [num_states, feature_dim]
                     The first state (index 0) is the target state to update
            depths: Optional tensor of shape [num_states] indicating the depth
                   of each state in the MCTS path
            
        Returns:
            Updated features tensor of the same shape.
            Only the target state (index 0) is modified.
        """
        current_features = features
        for layer in self.layers:
            current_features = layer(current_features, depths)
        return current_features