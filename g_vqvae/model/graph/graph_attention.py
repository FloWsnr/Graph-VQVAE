import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import math


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer inspired by Graphormer that performs full attention across the entire graph.
    
    This layer computes attention weights between all pairs of nodes in the graph,
    incorporating both node features and edge features in the attention computation.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of node features
    num_heads : int
        Number of attention heads
    edge_dim : int, optional
        Dimension of edge features. If None, edge features are not used.
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        num_heads: int = 8, 
        edge_dim: int = None, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.head_dim = out_channels // num_heads
        
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        
        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        
        # Edge feature projection if edge features are provided
        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(out_channels, out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, data: Data) -> Data:
        """
        Forward pass through the graph attention layer.
        
        Parameters
        ----------
        data : Data
            Input graph data containing:
            - x: node features of shape (num_nodes, hidden_dim)
            - edge_index: edge connectivity of shape (2, num_edges)
            - edge_attr: edge features of shape (num_edges, edge_dim) [optional]
            
        Returns
        -------
        Data
            Updated graph data with the same structure but updated node features
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        
        num_nodes = x.size(0)
        batch_size = 1  # Single graph
        
        # Store original features for residual connection (if dimensions match)
        if self.in_channels == self.out_channels:
            residual = x
        else:
            residual = None
        
        # Compute queries, keys, values
        q = self.q_proj(x)  # (num_nodes, hidden_dim)
        k = self.k_proj(x)  # (num_nodes, hidden_dim)
        v = self.v_proj(x)  # (num_nodes, hidden_dim)
        
        # Reshape for multi-head attention
        q = q.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, num_nodes, head_dim)
        k = k.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, num_nodes, head_dim)
        v = v.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, num_nodes, head_dim)
        
        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)  # (num_heads, num_nodes, num_nodes)
        
        # Add edge bias if edge features are provided
        if edge_attr is not None and self.edge_dim is not None:
            edge_bias = self._compute_edge_bias(edge_index, edge_attr, num_nodes)
            scores = scores + edge_bias
        
        # Apply attention mask (optional - for now we use full attention)
        # In practice, you might want to mask certain connections
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (num_heads, num_nodes, num_nodes)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.bmm(attn_weights, v)  # (num_heads, num_nodes, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 1).contiguous().view(num_nodes, self.out_channels)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Residual connection and layer norm
        if residual is not None:
            output = self.norm(output + residual)
        else:
            output = self.norm(output)
        
        # Return updated graph data
        return Data(
            x=output,
            edge_index=edge_index,
            edge_attr=edge_attr,
            **{k: v for k, v in data.__dict__.items() if k not in ['x', 'edge_index', 'edge_attr']}
        )
    
    def _compute_edge_bias(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute edge bias for attention scores based on edge features.
        
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge connectivity of shape (2, num_edges)
        edge_attr : torch.Tensor
            Edge features of shape (num_edges, edge_dim)
        num_nodes : int
            Number of nodes in the graph
            
        Returns
        -------
        torch.Tensor
            Edge bias of shape (num_heads, num_nodes, num_nodes)
        """
        # Project edge features to attention heads
        edge_bias_values = self.edge_proj(edge_attr)  # (num_edges, num_heads)
        
        # Create dense adjacency matrix with edge biases
        edge_bias = torch.zeros(self.num_heads, num_nodes, num_nodes, device=edge_attr.device)
        
        # Fill in the edge biases
        src, dst = edge_index[0], edge_index[1]
        for h in range(self.num_heads):
            edge_bias[h, src, dst] = edge_bias_values[:, h]
            # For undirected graphs, you might want to add symmetric bias:
            # edge_bias[h, dst, src] = edge_bias_values[:, h]
        
        return edge_bias


class MultiLayerGraphAttention(nn.Module):
    """
    Multi-layer graph attention network with residual connections.
    
    Parameters
    ----------
    in_channels : int
        Input dimension of node features
    hidden_dim : int
        Hidden dimension of node features
    num_layers : int
        Number of attention layers
    num_heads : int
        Number of attention heads per layer
    edge_dim : int, optional
        Dimension of edge features
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        edge_dim: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_dim = in_channels if i == 0 else hidden_dim
            self.layers.append(
                GraphAttentionLayer(layer_in_dim, hidden_dim, num_heads, edge_dim, dropout)
            )
        
    def forward(self, data: Data) -> Data:
        """
        Forward pass through multiple graph attention layers.
        
        Parameters
        ----------
        data : Data
            Input graph data
            
        Returns
        -------
        Data
            Output graph data with updated node features
        """
        for layer in self.layers:
            data = layer(data)
        return data


if __name__ == "__main__":
    # Test the implementation
    from torch_geometric.datasets import FakeDataset
    
    # Create fake data
    dataset = FakeDataset(num_graphs=1, avg_num_nodes=10, avg_degree=4, num_node_features=64, num_edge_features=8)
    data = dataset[0]
    
    # Create model
    model = GraphAttentionLayer(hidden_dim=64, num_heads=8, edge_dim=8)
    
    # Forward pass
    output = model(data)
    print(f"Input shape: {data.x.shape}")
    print(f"Output shape: {output.x.shape}")
    print("Graph attention layer test passed!")