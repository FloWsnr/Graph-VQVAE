"""
Tests for Graph Attention Layer

By: Claude Code
Date: 2025-06-04
"""

import pytest
import torch
from torch_geometric.data import Data

from g_vqvae.model.graph.graph_attention import GraphAttentionLayer, MultiLayerGraphAttention


class TestGraphAttentionLayer:
    """Test cases for GraphAttentionLayer"""
    
    def test_init(self):
        """Test initialization of GraphAttentionLayer"""
        layer = GraphAttentionLayer(hidden_dim=64, num_heads=8)
        assert layer.hidden_dim == 64
        assert layer.num_heads == 8
        assert layer.head_dim == 8
        assert layer.edge_dim is None
        
        # Test with edge features
        layer_with_edges = GraphAttentionLayer(hidden_dim=64, num_heads=8, edge_dim=16)
        assert layer_with_edges.edge_dim == 16
        assert hasattr(layer_with_edges, 'edge_proj')
        
    def test_init_invalid_dimensions(self):
        """Test initialization with invalid dimensions"""
        with pytest.raises(AssertionError):
            # hidden_dim not divisible by num_heads
            GraphAttentionLayer(hidden_dim=65, num_heads=8)
    
    def test_forward_basic(self, torch_geometric_graph):
        """Test basic forward pass without edge features"""
        # Adjust input features to match layer dimensions
        hidden_dim = 64
        data = torch_geometric_graph
        data.x = torch.randn(data.num_nodes, hidden_dim)
        
        layer = GraphAttentionLayer(hidden_dim=hidden_dim, num_heads=8)
        output = layer(data)
        
        # Check output properties
        assert isinstance(output, Data)
        assert output.x.shape == data.x.shape
        assert torch.allclose(output.edge_index, data.edge_index)
        assert not torch.allclose(output.x, data.x)  # Features should be updated
        
    def test_forward_with_edge_features(self, torch_geometric_graph):
        """Test forward pass with edge features"""
        hidden_dim = 64
        edge_dim = 16
        data = torch_geometric_graph
        data.x = torch.randn(data.num_nodes, hidden_dim)
        data.edge_attr = torch.randn(data.num_edges, edge_dim)
        
        layer = GraphAttentionLayer(hidden_dim=hidden_dim, num_heads=8, edge_dim=edge_dim)
        output = layer(data)
        
        # Check output properties
        assert isinstance(output, Data)
        assert output.x.shape == data.x.shape
        assert torch.allclose(output.edge_index, data.edge_index)
        assert torch.allclose(output.edge_attr, data.edge_attr)
        assert not torch.allclose(output.x, data.x)  # Features should be updated
        
    def test_forward_different_head_counts(self, torch_geometric_graph):
        """Test forward pass with different numbers of attention heads"""
        hidden_dim = 64
        data = torch_geometric_graph
        data.x = torch.randn(data.num_nodes, hidden_dim)
        
        for num_heads in [1, 2, 4, 8, 16]:
            layer = GraphAttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads)
            output = layer(data)
            assert output.x.shape == data.x.shape
            
    def test_edge_bias_computation(self, torch_geometric_graph):
        """Test edge bias computation"""
        hidden_dim = 64
        edge_dim = 16
        num_heads = 8
        
        data = torch_geometric_graph
        data.x = torch.randn(data.num_nodes, hidden_dim)
        data.edge_attr = torch.randn(data.num_edges, edge_dim)
        
        layer = GraphAttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, edge_dim=edge_dim)
        
        # Test the private method
        edge_bias = layer._compute_edge_bias(data.edge_index, data.edge_attr, data.num_nodes)
        
        expected_shape = (num_heads, data.num_nodes, data.num_nodes)
        assert edge_bias.shape == expected_shape
        
        # Check that edge bias is non-zero where edges exist
        src, dst = data.edge_index[0], data.edge_index[1]
        for h in range(num_heads):
            for i in range(data.num_edges):
                assert edge_bias[h, src[i], dst[i]] != 0
                
    def test_gradient_flow(self, torch_geometric_graph):
        """Test that gradients flow through the layer"""
        hidden_dim = 64
        data = torch_geometric_graph
        data.x = torch.randn(data.num_nodes, hidden_dim, requires_grad=True)
        
        layer = GraphAttentionLayer(hidden_dim=hidden_dim, num_heads=8)
        output = layer(data)
        
        # Compute a simple loss
        loss = output.x.sum()
        loss.backward()
        
        # Check that input gradients exist
        assert data.x.grad is not None
        assert not torch.allclose(data.x.grad, torch.zeros_like(data.x.grad))
        
    def test_dropout_effect(self, torch_geometric_graph):
        """Test that dropout has an effect during training"""
        hidden_dim = 64
        data = torch_geometric_graph
        data.x = torch.randn(data.num_nodes, hidden_dim)
        
        layer = GraphAttentionLayer(hidden_dim=hidden_dim, num_heads=8, dropout=0.5)
        
        # Set to training mode
        layer.train()
        output1 = layer(data)
        output2 = layer(data)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output1.x, output2.x)
        
        # Set to eval mode
        layer.eval()
        output3 = layer(data)
        output4 = layer(data)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output3.x, output4.x)


class TestMultiLayerGraphAttention:
    """Test cases for MultiLayerGraphAttention"""
    
    def test_init(self):
        """Test initialization of MultiLayerGraphAttention"""
        model = MultiLayerGraphAttention(hidden_dim=64, num_layers=3, num_heads=8)
        assert len(model.layers) == 3
        assert all(isinstance(layer, GraphAttentionLayer) for layer in model.layers)
        
    def test_forward(self, torch_geometric_graph):
        """Test forward pass through multiple layers"""
        hidden_dim = 64
        data = torch_geometric_graph
        data.x = torch.randn(data.num_nodes, hidden_dim)
        
        model = MultiLayerGraphAttention(hidden_dim=hidden_dim, num_layers=3, num_heads=8)
        output = model(data)
        
        # Check output properties
        assert isinstance(output, Data)
        assert output.x.shape == data.x.shape
        assert torch.allclose(output.edge_index, data.edge_index)
        assert not torch.allclose(output.x, data.x)  # Features should be updated
        
    def test_forward_with_edge_features(self, torch_geometric_graph):
        """Test multi-layer forward pass with edge features"""
        hidden_dim = 64
        edge_dim = 16
        data = torch_geometric_graph
        data.x = torch.randn(data.num_nodes, hidden_dim)
        data.edge_attr = torch.randn(data.num_edges, edge_dim)
        
        model = MultiLayerGraphAttention(
            hidden_dim=hidden_dim, 
            num_layers=2, 
            num_heads=8, 
            edge_dim=edge_dim
        )
        output = model(data)
        
        # Check output properties
        assert isinstance(output, Data)
        assert output.x.shape == data.x.shape
        assert torch.allclose(output.edge_index, data.edge_index)
        assert torch.allclose(output.edge_attr, data.edge_attr)


class TestGraphAttentionIntegration:
    """Integration tests for graph attention components"""
    
    def test_attention_with_various_graph_sizes(self):
        """Test attention layer with different graph sizes"""
        hidden_dim = 32
        layer = GraphAttentionLayer(hidden_dim=hidden_dim, num_heads=4)
        
        # Test with different graph sizes
        for num_nodes in [5, 10, 20, 50]:
            # Create simple graph
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            if num_nodes > 3:
                # Add more edges for larger graphs
                additional_edges = torch.randint(0, num_nodes, (2, num_nodes))
                edge_index = torch.cat([edge_index, additional_edges], dim=1)
            
            x = torch.randn(num_nodes, hidden_dim)
            data = Data(x=x, edge_index=edge_index)
            
            output = layer(data)
            assert output.x.shape == (num_nodes, hidden_dim)
            
    def test_memory_efficiency(self, torch_geometric_graph):
        """Test memory usage is reasonable"""
        hidden_dim = 64
        data = torch_geometric_graph
        data.x = torch.randn(data.num_nodes, hidden_dim)
        
        layer = GraphAttentionLayer(hidden_dim=hidden_dim, num_heads=8)
        
        # Check model parameters count
        total_params = sum(p.numel() for p in layer.parameters())
        expected_params_range = (hidden_dim * hidden_dim * 3, hidden_dim * hidden_dim * 10)  # Rough estimate
        assert expected_params_range[0] <= total_params <= expected_params_range[1]
        
    def test_reproducibility(self, torch_geometric_graph):
        """Test that results are reproducible with same seed"""
        hidden_dim = 64
        data = torch_geometric_graph
        data.x = torch.randn(data.num_nodes, hidden_dim)
        
        # Set seed and run
        torch.manual_seed(42)
        layer1 = GraphAttentionLayer(hidden_dim=hidden_dim, num_heads=8)
        layer1.eval()
        output1 = layer1(data)
        
        # Reset seed and run again
        torch.manual_seed(42)
        layer2 = GraphAttentionLayer(hidden_dim=hidden_dim, num_heads=8)
        layer2.eval()
        output2 = layer2(data)
        
        # Results should be identical
        assert torch.allclose(output1.x, output2.x, atol=1e-6)