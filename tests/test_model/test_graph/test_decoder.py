import torch
import pytest
from torch_geometric.data import Data
from g_vqvae.model.graph.decoder import GraphDecoder


def test_decoder_forward_gcn():
    """Test GraphDecoder forward pass with GCN layers."""
    # Create encoded features (simulating encoder output)
    num_encoded_nodes = 5
    codebook_dim = 64
    num_original_nodes = 10
    
    encoded_features = torch.randn(num_encoded_nodes, codebook_dim)
    
    decoder = GraphDecoder(
        codebook_dim=codebook_dim,
        hidden_dim=64,
        out_channels=32,
        layer_type="gcn"
    )
    
    output = decoder(encoded_features, num_original_nodes)
    
    # Check output
    assert isinstance(output, Data)
    assert output.x.size(0) == num_original_nodes  # Should restore original number of nodes
    assert output.x.size(1) == 32  # out_channels
    assert hasattr(output, 'edge_index')
    assert hasattr(output, 'edge_attr')
    
    # Check that it creates a fully connected graph (minus self-loops)
    expected_num_edges = num_original_nodes * (num_original_nodes - 1)
    assert output.edge_index.size(1) == expected_num_edges


def test_decoder_forward_gat():
    """Test GraphDecoder forward pass with GAT layers."""
    num_encoded_nodes = 5
    codebook_dim = 64
    num_original_nodes = 10
    
    encoded_features = torch.randn(num_encoded_nodes, codebook_dim)
    
    decoder = GraphDecoder(
        codebook_dim=codebook_dim,
        hidden_dim=64,
        out_channels=32,
        layer_type="gat"
    )
    
    output = decoder(encoded_features, num_original_nodes)
    
    assert isinstance(output, Data)
    assert output.x.size(0) == num_original_nodes
    assert output.x.size(1) == 32


def test_decoder_forward_attention():
    """Test GraphDecoder forward pass with Graph Attention layers."""
    num_encoded_nodes = 5
    codebook_dim = 64
    num_original_nodes = 10
    
    encoded_features = torch.randn(num_encoded_nodes, codebook_dim)
    
    decoder = GraphDecoder(
        codebook_dim=codebook_dim,
        hidden_dim=64,
        out_channels=32,
        layer_type="attention"
    )
    
    output = decoder(encoded_features, num_original_nodes)
    
    assert isinstance(output, Data)
    assert output.x.size(0) == num_original_nodes
    assert output.x.size(1) == 32


def test_edge_index_reconstruction():
    """Test that decoder creates proper edge indices."""
    num_encoded_nodes = 3
    codebook_dim = 64
    num_original_nodes = 5
    
    encoded_features = torch.randn(num_encoded_nodes, codebook_dim)
    
    decoder = GraphDecoder(
        codebook_dim=codebook_dim,
        hidden_dim=32,
        out_channels=16,
        layer_type="gcn"
    )
    
    output = decoder(encoded_features, num_original_nodes)
    
    # Check edge index properties
    edge_index = output.edge_index
    assert edge_index.size(0) == 2  # Should have source and target indices
    assert torch.all(edge_index >= 0)  # All indices should be non-negative
    assert torch.all(edge_index < num_original_nodes)  # All indices should be valid
    
    # Check no self-loops
    src, dst = edge_index[0], edge_index[1]
    assert torch.all(src != dst)