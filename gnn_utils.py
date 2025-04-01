import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, feature_dim):
        super(GNNLayer, self).__init__()
        self.feature_dim = feature_dim
        
        # Attention mechanism
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
        combined = torch.cat([target_feat, source_feat], dim=1)
        return torch.sigmoid(self.attention(combined))
    
    def forward(self, features):
        if features.size(0) <= 1:
            return features
            
        # The first feature vector is the target state
        target_feat = features[0:1]
        
        # Process messages from all other states
        path_states = features[1:]
        
        if path_states.size(0) == 0:
            return features
        
        # Compute attention weights for each path state
        attention_weights = []
        for i in range(path_states.size(0)):
            source_feat = path_states[i:i+1]
            attn = self.compute_attention(target_feat, source_feat)
            attention_weights.append(attn)
        
        # Stack attention weights
        attention_weights = torch.cat(attention_weights, dim=0)
        
        # Normalize attention weights
        if attention_weights.sum() > 0:
            attention_weights = attention_weights / attention_weights.sum()
        
        # Apply attention weights to path states
        weighted_path_feats = path_states * attention_weights.view(-1, 1)
        
        # Aggregate weighted features
        aggregated_path_info = weighted_path_feats.sum(dim=0, keepdim=True)
        
        # Update target state with aggregated path information using gating
        combined = torch.cat([target_feat, aggregated_path_info], dim=1)
        gate_value = self.gate(combined)
        update_value = self.update_net(combined)
        updated_target = target_feat + gate_value * update_value
        
        # Return updated target state along with original path states
        return torch.cat([updated_target, path_states], dim=0)

class GNNProcessor(nn.Module):
    def __init__(self, feature_dim, num_layers=2):
        super(GNNProcessor, self).__init__()
        self.layers = nn.ModuleList([GNNLayer(feature_dim) for _ in range(num_layers)])
    
    def forward(self, features):
        current_features = features
        for layer in self.layers:
            current_features = layer(current_features)
        return current_features

class PolicyValueGNN(nn.Module):
    """
    GNN that enhances the hidden representation from the neural network.
    It takes the hidden feature map and enhances it, then we pass the
    enhanced representation to the original policy/value heads.
    """
    def __init__(self, feature_dim, num_layers=2):
        super(PolicyValueGNN, self).__init__()
        self.feature_dim = feature_dim
        
        # GNN layers to enhance the feature representation
        self.layers = nn.ModuleList([GNNLayer(feature_dim) for _ in range(num_layers)])
        
        # Final transformation to ensure output dimension matches input
        self.output_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, features):
        """Process the hidden representation to produce an enhanced representation"""
        # Apply GNN layers to enhance features
        enhanced = features.clone()
        for layer in self.layers:
            enhanced = layer(enhanced)
        
        # Final transformation to ensure dimension compatibility
        enhanced_features = self.output_transform(enhanced)
        
        return enhanced_features