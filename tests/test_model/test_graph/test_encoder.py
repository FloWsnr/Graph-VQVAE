import torch
import pytest
from torch_geometric.data import Data
from g_vqvae.model.graph.encoder import GraphEncoder


def test_encoder_forward_gcn(torch_geometric_graph):
    """Test GraphEncoder forward pass with GCN layers."""
    in_channels = torch_geometric_graph.num_node_features
    
    encoder = GraphEncoder(
        in_channels=in_channels,
        hidden_dim=64,
        layer_type="gcn",
        pooling_type="SAG"
    )
    
    output = encoder(torch_geometric_graph)
    
    # Check output is Data object
    assert isinstance(output, Data)
    assert hasattr(output, 'x')
    assert hasattr(output, 'edge_index')
    
    # Check that pooling reduced the number of nodes
    assert output.x.size(0) <= torch_geometric_graph.x.size(0)
    assert output.x.size(1) == 64  # hidden_dim


def test_encoder_forward_gat(torch_geometric_graph):
    """Test GraphEncoder forward pass with GAT layers."""
    in_channels = torch_geometric_graph.num_node_features
    
    encoder = GraphEncoder(
        in_channels=in_channels,
        hidden_dim=64,
        layer_type="gat",
        pooling_type="SAG"
    )
    
    output = encoder(torch_geometric_graph)
    
    assert isinstance(output, Data)
    assert output.x.size(0) <= torch_geometric_graph.x.size(0)
    assert output.x.size(1) == 64


def test_encoder_forward_attention(torch_geometric_graph):
    """Test GraphEncoder forward pass with Graph Attention layers."""
    in_channels = torch_geometric_graph.num_node_features
    
    encoder = GraphEncoder(
        in_channels=in_channels,
        hidden_dim=64,
        layer_type="attention",
        pooling_type="SAG"
    )
    
    output = encoder(torch_geometric_graph)
    
    assert isinstance(output, Data)
    assert output.x.size(0) <= torch_geometric_graph.x.size(0)
    assert output.x.size(1) == 64