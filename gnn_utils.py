from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyValueGNN(nn.Module):
    def __init__(self, feature_dim, num_layers=10, heads=1, dropout=0.0):
        super().__init__()

        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            if layer_idx % 2 == 0:
                # Even‐indexed layer: use concat=True
                #   => out_channels is per‐head size, so heads * out_channels = feature_dim
                self.layers.append(
                    GATConv(
                        in_channels=feature_dim,
                        out_channels=feature_dim // heads,
                        heads=heads,
                        concat=True,
                        dropout=dropout
                    )
                )
            else:
                # Odd‐indexed layer: use concat=False
                #   => out_channels must be the full feature_dim
                self.layers.append(
                    GATConv(
                        in_channels=feature_dim,
                        out_channels=feature_dim,  # final dim = feature_dim
                        heads=heads,
                        concat=False,
                        dropout=dropout
                    )
                )

        # LayerNorm on the [num_nodes, feature_dim] tensor after each residual sum
        self.norms = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(num_layers)])

    def forward(self, x, edge_index):
        h = x
        for conv, norm in zip(self.layers, self.norms):
            h_new = conv(h, edge_index)
            h     = F.relu(norm(h_new + h))
        return h
