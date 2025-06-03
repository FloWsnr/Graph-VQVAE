import torch
from torch_geometric.data import Data
from g_vqvae.model.graph_vq import GraphVectorQuantizer


def test_graph_vector_quantizer_initialization():
    """
    Test the initialization of the GraphVectorQuantizer.
    """
    codebook_size = 512
    codebook_dim = 64
    commitment_cost = 0.25

    quantizer = GraphVectorQuantizer(
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
        commitment_cost=commitment_cost,
    )

    assert quantizer.codebook_size == codebook_size
    assert quantizer.codebook_dim == codebook_dim
    assert quantizer.commitment_cost == commitment_cost
    assert quantizer.embedding.weight.shape == (codebook_size, codebook_dim)


def test_graph_vector_quantizer_forward(torch_geometric_graph):
    """
    Test the forward pass of the GraphVectorQuantizer.
    """
    codebook_dim = 64
    codebook_size = 512

    quantizer = GraphVectorQuantizer(codebook_size, codebook_dim)

    # Create input data with node features in codebook dimension
    data = Data(
        x=torch.randn(torch_geometric_graph.num_nodes, codebook_dim),
        edge_index=torch_geometric_graph.edge_index,
    )

    quantized_data, loss, indices = quantizer(data)

    # Check shapes and types
    assert quantized_data.x.shape == data.x.shape
    assert indices.shape == (data.num_nodes,)
    assert isinstance(loss, torch.Tensor)

    # Check that edge structure is preserved
    assert torch.allclose(quantized_data.edge_index, data.edge_index)

    # Check that quantized values are from codebook
    for i, idx in enumerate(indices):
        assert torch.allclose(quantized_data.x[i], quantizer.embedding.weight[idx])


def test_graph_vector_quantizer_straight_through(torch_geometric_graph):
    """
    Test that the straight-through estimator is working correctly.
    """
    codebook_dim = 64
    codebook_size = 512

    quantizer = GraphVectorQuantizer(codebook_size, codebook_dim)

    # Create input data with node features in codebook dimension
    data = Data(
        x=torch.randn(torch_geometric_graph.num_nodes, codebook_dim),
        edge_index=torch_geometric_graph.edge_index,
    )

    quantized_data, loss, indices = quantizer(data)

    # Check that gradients can flow through the straight-through estimator
    loss.backward()
    assert data.x.grad is not None
    assert quantizer.embedding.weight.grad is not None


def test_graph_vector_quantizer_loss(torch_geometric_graph):
    """
    Test that the commitment loss is computed correctly.
    """
    codebook_dim = 64
    codebook_size = 512
    commitment_cost = 0.25

    quantizer = GraphVectorQuantizer(codebook_size, codebook_dim, commitment_cost)

    # Create input data with node features in codebook dimension
    data = Data(
        x=torch.randn(torch_geometric_graph.num_nodes, codebook_dim),
        edge_index=torch_geometric_graph.edge_index,
    )

    quantized_data, loss, indices = quantizer(data)

    # Compute expected loss components
    codebook_loss = torch.mean((quantized_data.x.detach() - data.x) ** 2)
    commitment_loss = commitment_cost * torch.mean(
        (quantized_data.x - data.x.detach()) ** 2
    )
    expected_loss = codebook_loss + commitment_loss

    assert torch.allclose(loss, expected_loss)


def test_graph_vector_quantizer_preserves_attributes(torch_geometric_graph):
    """
    Test that the quantizer preserves additional graph attributes.
    """
    codebook_dim = 64
    codebook_size = 512

    quantizer = GraphVectorQuantizer(codebook_size, codebook_dim)

    # Create input data with additional attributes
    data = Data(
        x=torch.randn(torch_geometric_graph.num_nodes, codebook_dim),
        edge_index=torch_geometric_graph.edge_index,
        edge_attr=torch.randn(torch_geometric_graph.edge_index.shape[1], 3),
        y=torch.randn(1),
    )

    quantized_data, loss, indices = quantizer(data)

    # Check that additional attributes are preserved
    assert hasattr(quantized_data, "edge_attr")
    assert hasattr(quantized_data, "y")
    assert torch.allclose(quantized_data.edge_attr, data.edge_attr)
    assert torch.allclose(quantized_data.y, data.y)
