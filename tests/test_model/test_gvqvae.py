import torch
from torch_geometric.data import Data
from g_vqvae.model.graph.graph_vqvae import G_VQVAE


def test_gnn_encoder(torch_geometric_graph):
    """
    Test the GNN encoder component.
    """
    in_channels = torch_geometric_graph.num_node_features
    hidden_dim = 256
    codebook_size = 512
    codebook_dim = 256

    model = G_VQVAE(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
    )

    quantized_data, loss, indices = model.encode(torch_geometric_graph)

    # Check shapes
    assert quantized_data.x.shape == (torch_geometric_graph.num_nodes, codebook_dim)
    assert isinstance(loss, torch.Tensor)
    assert indices.shape == (torch_geometric_graph.num_nodes,)
    # check edges are preserved
    assert torch.allclose(torch_geometric_graph.edge_index, quantized_data.edge_index)


def test_gnn_decoder(torch_geometric_graph):
    """
    Test the GNN decoder component.
    """
    in_channels = torch_geometric_graph.num_node_features
    hidden_dim = 256
    codebook_size = 512
    codebook_dim = 256

    model = G_VQVAE(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
    )

    # Create a quantized data object
    quantized_data = Data(
        x=torch.randn(torch_geometric_graph.num_nodes, codebook_dim),
        edge_index=torch_geometric_graph.edge_index,
    )

    # Decode
    reconstructed_data = model.decode(quantized_data)

    # Check shape matches input
    assert reconstructed_data.x.shape == (torch_geometric_graph.num_nodes, in_channels)
    # Check edges are preserved
    assert torch.allclose(
        torch_geometric_graph.edge_index, reconstructed_data.edge_index
    )


def test_gnn_vqvae(torch_geometric_graph):
    """
    Test the full GNN VQVAE model.
    """
    in_channels = torch_geometric_graph.num_node_features
    hidden_dim = 256
    codebook_size = 512
    codebook_dim = 256

    model = G_VQVAE(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
    )

    # Forward pass
    reconstructed_data, loss, indices = model(torch_geometric_graph)

    # Check shapes and types
    assert reconstructed_data.x.shape == torch_geometric_graph.x.shape
    assert isinstance(loss, torch.Tensor)
    assert indices.shape == (torch_geometric_graph.num_nodes,)
    # Check edges are preserved
    assert torch.allclose(
        torch_geometric_graph.edge_index, reconstructed_data.edge_index
    )

    # Check that reconstruction is different from input (due to quantization)
    assert not torch.allclose(reconstructed_data.x, torch_geometric_graph.x)
