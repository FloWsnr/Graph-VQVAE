import torch
import pytest
from torch_geometric.data import Data
from g_vqvae.model.graph.graph_vqvae import G_VQVAE


def test_gnn_vqvae_forward_gcn(torch_geometric_graph):
    """Test the full GNN VQVAE model with GCN layers."""
    in_channels = torch_geometric_graph.num_node_features
    
    model = G_VQVAE(
        in_channels=in_channels,
        hidden_dim=64,
        codebook_size=256,
        codebook_dim=64,
        encoder_config={"layer_type": "gcn", "pooling_type": "SAG"},
        decoder_config={"layer_type": "gcn"}
    )
    
    # Forward pass
    reconstructed_data, loss, indices = model(torch_geometric_graph)
    
    # Check shapes and types
    assert isinstance(reconstructed_data, Data)
    assert reconstructed_data.x.size(-1) == torch_geometric_graph.x.size(-1)  # Feature dimension should match
    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1  # Loss should be scalar
    assert isinstance(indices, torch.Tensor)
    
    # Check that reconstruction is different from input (due to quantization)
    assert not torch.allclose(reconstructed_data.x, torch_geometric_graph.x)


def test_gnn_vqvae_forward_gat(torch_geometric_graph):
    """Test the full GNN VQVAE model with GAT layers."""
    in_channels = torch_geometric_graph.num_node_features
    
    model = G_VQVAE(
        in_channels=in_channels,
        hidden_dim=64,
        codebook_dim=64,
        encoder_config={"layer_type": "gat", "pooling_type": "SAG"},
        decoder_config={"layer_type": "gat"}
    )
    
    # Forward pass
    reconstructed_data, loss, indices = model(torch_geometric_graph)
    
    # Check shapes and types
    assert isinstance(reconstructed_data, Data)
    assert reconstructed_data.x.size(-1) == torch_geometric_graph.x.size(-1)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(indices, torch.Tensor)


def test_gnn_vqvae_forward_attention(torch_geometric_graph):
    """Test the full GNN VQVAE model with Graph Attention layers."""
    in_channels = torch_geometric_graph.num_node_features
    
    model = G_VQVAE(
        in_channels=in_channels,
        hidden_dim=64,
        codebook_dim=64,
        encoder_config={"layer_type": "attention", "pooling_type": "SAG"},
        decoder_config={"layer_type": "attention"}
    )
    
    # Forward pass
    reconstructed_data, loss, indices = model(torch_geometric_graph)
    
    # Check shapes and types
    assert isinstance(reconstructed_data, Data)
    assert reconstructed_data.x.size(-1) == torch_geometric_graph.x.size(-1)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(indices, torch.Tensor)


def test_gradient_flow(torch_geometric_graph):
    """Test that gradients flow through the model."""
    in_channels = torch_geometric_graph.num_node_features
    torch_geometric_graph.x.requires_grad_(True)
    
    model = G_VQVAE(in_channels=in_channels, hidden_dim=64, codebook_dim=64)
    reconstructed_data, loss, indices = model(torch_geometric_graph)
    
    # Compute a simple loss
    total_loss = loss + reconstructed_data.x.sum()
    total_loss.backward()
    
    # Check that input gradients exist
    assert torch_geometric_graph.x.grad is not None
    assert not torch.allclose(torch_geometric_graph.x.grad, torch.zeros_like(torch_geometric_graph.x.grad))


def test_different_hidden_dimensions(torch_geometric_graph):
    """Test model with different hidden dimensions."""
    in_channels = torch_geometric_graph.num_node_features
    
    for hidden_dim in [32, 64, 128]:
        model = G_VQVAE(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            codebook_dim=hidden_dim
        )
        
        reconstructed_data, loss, indices = model(torch_geometric_graph)
        assert reconstructed_data.x.size(-1) == in_channels
        assert isinstance(loss, torch.Tensor)


def test_different_codebook_sizes(torch_geometric_graph):
    """Test model with different codebook sizes."""
    in_channels = torch_geometric_graph.num_node_features
    
    for codebook_size in [64, 128, 256]:
        model = G_VQVAE(
            in_channels=in_channels,
            hidden_dim=64,
            codebook_dim=64,
            codebook_size=codebook_size
        )
        
        reconstructed_data, loss, indices = model(torch_geometric_graph)
        assert reconstructed_data.x.size(-1) == in_channels
        assert indices.max() < codebook_size  # All indices should be valid


def test_model_deterministic():
    """Test that model produces deterministic results with same seed."""
    from torch_geometric.datasets import FakeDataset
    
    dataset = FakeDataset(num_graphs=1, avg_num_nodes=10, avg_degree=4)
    data = dataset[0]
    in_channels = data.x.size(1)
    
    # First run
    torch.manual_seed(42)
    model1 = G_VQVAE(in_channels=in_channels, hidden_dim=64, codebook_dim=64)
    model1.eval()
    with torch.no_grad():
        output1, loss1, indices1 = model1(data)
    
    # Second run with same seed
    torch.manual_seed(42)
    model2 = G_VQVAE(in_channels=in_channels, hidden_dim=64, codebook_dim=64)
    model2.eval()
    with torch.no_grad():
        output2, loss2, indices2 = model2(data)
    
    # Results should be identical
    assert torch.allclose(output1.x, output2.x, atol=1e-6)
    assert torch.allclose(loss1, loss2, atol=1e-6)
    assert torch.allclose(indices1, indices2)