import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
from typing import Literal

from g_vqvae.model.graph.graph_attention import GraphAttentionLayer


class GraphEncoder(nn.Module):
    """
    Encoder for the G_VQVAE.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        layer_type: Literal["gcn", "gat", "attention"] = "gcn",
        pooling_type: Literal["edge", "SAG"] = "edge",
    ):
        super().__init__()

        if layer_type == "gcn":
            self.conv1 = gnn.GCNConv(in_channels, hidden_dim)
            self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim)
            self.relu = nn.ReLU()
        elif layer_type == "gat":
            self.conv1 = gnn.GATConv(in_channels, hidden_dim)
            self.conv2 = gnn.GATConv(hidden_dim, hidden_dim)
            self.relu = nn.ReLU()
        elif layer_type == "attention":
            self.conv1 = GraphAttentionLayer(in_channels, hidden_dim)
            self.conv2 = GraphAttentionLayer(hidden_dim, hidden_dim)
            self.relu = nn.ReLU()
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")

        if pooling_type == "SAG":
            self.pool = gnn.SAGPooling(in_channels=hidden_dim, ratio=0.5)
        else:
            raise ValueError(f"Invalid pooling type: {pooling_type}")

    def forward(self, data: Data) -> Data:
        """
        Forward pass through the encoder.
        """
        x = data.x
        edge_index = data.edge_index
        
        # First convolution
        if hasattr(self.conv1, 'forward') and isinstance(self.conv1, GraphAttentionLayer):
            temp_data = Data(x=x, edge_index=edge_index)
            temp_data = self.conv1(temp_data)
            x = temp_data.x
        else:
            x = self.conv1(x, edge_index)
        x = self.relu(x)
        
        # Pooling
        x, edge_index, edge_attr, batch, perm, score = self.pool(x, edge_index)
        
        # Second convolution
        if hasattr(self.conv2, 'forward') and isinstance(self.conv2, GraphAttentionLayer):
            temp_data = Data(x=x, edge_index=edge_index)
            temp_data = self.conv2(temp_data)
            x = temp_data.x
        else:
            x = self.conv2(x, edge_index)
        x = self.relu(x)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
