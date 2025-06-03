import torch
from g_vqvae.model.image.vqvae import (
    ResidualBlock,
    VectorQuantizer,
    Encoder,
    Decoder,
    VQVAE,
)


def test_residual_block():
    batch_size = 2
    channels = 64
    height = 32
    width = 32

    block = ResidualBlock(channels)
    x = torch.randn(batch_size, channels, height, width)

    output = block(x)
    assert output.shape == (batch_size, channels, height, width)


def test_vector_quantizer():
    batch_size = 2
    height = 16
    width = 16
    codebook_dim = 64
    codebook_size = 512

    quantizer = VectorQuantizer(codebook_size, codebook_dim)
    z = torch.randn(batch_size, height, width, codebook_dim)

    z_q, loss, indices = quantizer(z)
    assert z_q.shape == z.shape
    assert isinstance(loss, torch.Tensor)
    assert indices.shape == (batch_size * height * width,)


def test_vqvae_tokenizer():
    batch_size = 2
    height = 32
    width = 32
    in_channels = 3
    hidden_dim = 256

    tokenizer = Encoder(in_channels, hidden_dim)
    x = torch.randn(batch_size, in_channels, height, width)

    z = tokenizer(x)
    # After two stride-2 convolutions, spatial dimensions are reduced by 4
    assert z.shape == (batch_size, hidden_dim, height // 4, width // 4)


def test_vqvae_detokenizer():
    batch_size = 2
    height = 16
    width = 16
    hidden_dim = 256
    out_channels = 3

    detokenizer = Decoder(out_channels, hidden_dim)
    z_q = torch.randn(batch_size, hidden_dim, height, width)

    x_recon = detokenizer(z_q)
    # After two stride-2 transposed convolutions, spatial dimensions are increased by 4
    assert x_recon.shape == (batch_size, out_channels, height * 4, width * 4)


def test_vqvae():
    batch_size = 2
    height = 32
    width = 32
    in_channels = 3
    hidden_dim = 256
    codebook_size = 512
    codebook_dim = 64

    vqvae = VQVAE(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
    )
    x = torch.randn(batch_size, in_channels, height, width)

    x_recon, loss, indices = vqvae(x)
    assert x_recon.shape == x.shape
    assert isinstance(loss, torch.Tensor)
    assert indices.shape == (batch_size * height // 4 * width // 4,)
