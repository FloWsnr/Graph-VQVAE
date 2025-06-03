import torch
from vqvae.model.vq import VectorQuantizer


def test_vector_quantizer_initialization():
    """
    Test the initialization of the VectorQuantizer.
    """
    codebook_size = 512
    codebook_dim = 64
    commitment_cost = 0.25

    quantizer = VectorQuantizer(
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
        commitment_cost=commitment_cost,
    )

    assert quantizer.codebook_size == codebook_size
    assert quantizer.codebook_dim == codebook_dim
    assert quantizer.commitment_cost == commitment_cost
    assert quantizer.embedding.weight.shape == (codebook_size, codebook_dim)


def test_vector_quantizer_forward():
    """
    Test the forward pass of the VectorQuantizer.
    """
    batch_size = 2
    height = 16
    width = 16
    codebook_dim = 64
    codebook_size = 512

    quantizer = VectorQuantizer(codebook_size, codebook_dim)
    z = torch.randn(batch_size, height, width, codebook_dim)

    z_q, loss, indices = quantizer(z)

    # Check shapes
    assert z_q.shape == z.shape
    assert indices.shape == (batch_size * height * width,)
    assert isinstance(loss, torch.Tensor)


def test_vector_quantizer_straight_through():
    """
    Test that the straight-through estimator is working correctly.
    """
    batch_size = 2
    height = 16
    width = 16
    codebook_dim = 64
    codebook_size = 512

    quantizer = VectorQuantizer(codebook_size, codebook_dim)
    z = torch.randn(batch_size, height, width, codebook_dim, requires_grad=True)

    z_q, loss, indices = quantizer(z)

    # Check that gradients can flow through the straight-through estimator
    loss.backward()
    assert z.grad is not None
    assert quantizer.embedding.weight.grad is not None


def test_vector_quantizer_loss():
    """
    Test that the commitment loss is computed correctly.
    """
    batch_size = 2
    height = 16
    width = 16
    codebook_dim = 64
    codebook_size = 512
    commitment_cost = 0.25

    quantizer = VectorQuantizer(codebook_size, codebook_dim, commitment_cost)
    z = torch.randn(batch_size, height, width, codebook_dim)

    z_q, loss, indices = quantizer(z)

    # Compute expected loss components
    codebook_loss = torch.mean((z_q.detach() - z) ** 2)
    commitment_loss = commitment_cost * torch.mean((z_q - z.detach()) ** 2)
    expected_loss = codebook_loss + commitment_loss

    assert torch.allclose(loss, expected_loss)
