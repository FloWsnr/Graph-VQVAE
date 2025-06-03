import torch
import torch.nn as nn
from vqvae.model.vq import VectorQuantizer


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutions and a skip connection.

    Parameters
    ----------
    channels : int
        Number of input and output channels
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.block(x)
        return x + residual


class Encoder(nn.Module):
    """
    VQ-VAE Tokenizer that encodes input into latent representation.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    hidden_dim : int
        Hidden dimension for the encoder conv layers
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # First strided conv
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Second strided conv
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            # Two residual blocks
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the VQ-VAE tokenizer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape (batch_size, channels, height, width)
        """
        return self.encoder(x)


class Decoder(nn.Module):
    """
    VQ-VAE Detokenizer that decodes discrete latent codes back to the original space.

    Parameters
    ----------
    out_channels : int
        Number of output channels
    hidden_dim : int
        Hidden dimension for the decoder
    """

    def __init__(
        self,
        out_channels: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Decoder
        self.decoder = nn.Sequential(
            # Two residual blocks
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            # First transposed conv
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            # Second transposed conv
            nn.ConvTranspose2d(
                hidden_dim, out_channels, kernel_size=4, stride=2, padding=1
            ),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the VQ-VAE detokenizer.

        Parameters
        ----------
        z_q : torch.Tensor
            Quantized tensor of shape (batch_size, channels, time, height, width)

        Returns
        -------
        torch.Tensor
            Reconstructed tensor of shape (batch_size, channels, time, height, width)
        """
        return self.decoder(z_q)


class VQVAE(nn.Module):
    """
    Complete VQ-VAE model that combines encoder and decoder.

    Parameters
    ----------
    in_channels : int
        Number of input channels (physical fields)
    hidden_dim : int
        Hidden dimension for encoder/decoder
        Convolutional layer dimensions (channels)
    codebook_size : int
        Number of embeddings in the codebook
        This is the number of discrete latent codes
    codebook_dim : int
        Dimension of each embedding vector
        This is the dimension of each latent code

    commitment_cost : float
        Commitment cost for the codebook loss
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        codebook_size: int = 512,
        codebook_dim: int = 256,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        num_fields = in_channels
        input_channels = in_channels
        self.encoder = Encoder(
            in_channels=input_channels,
            hidden_dim=hidden_dim,
        )

        # Projection layers for codebook
        self.to_codebook = nn.Conv2d(hidden_dim, codebook_dim, kernel_size=1)
        self.from_codebook = nn.Conv2d(codebook_dim, hidden_dim, kernel_size=1)

        # Vector Quantizer
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            commitment_cost=commitment_cost,
        )

        self.decoder = Decoder(
            out_channels=num_fields,
            hidden_dim=hidden_dim,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor.
        """
        z = self.encoder(x)
        z = self.to_codebook(z)
        z_q, loss, indices = self.quantizer(z)
        return z_q, loss, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode the quantized tensor.
        """
        z_q = self.from_codebook(z_q)
        x_recon = self.decoder(z_q)
        return x_recon

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VQ-VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Reconstructed tensor
            - Codebook loss
            - Encoding indices
        """
        # Encode
        z_q, loss, indices = self.encode(x)
        # Decode
        x_recon = self.decode(z_q)

        return x_recon, loss, indices
