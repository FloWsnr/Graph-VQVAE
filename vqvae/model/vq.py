import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module that maps continuous inputs to discrete codes.

    Parameters
    ----------
    codebook_size : int
        Number of embeddings in the codebook
    codebook_dim : int
        Dimension of each embedding vector
    commitment_cost : float
        Commitment cost for the codebook loss
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost

        # Initialize codebook
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        self.embedding.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the vector quantizer.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Quantized tensor
            - Codebook loss
            - Encoding indices
        """
        # Reshape input to (batch_size * height * width)
        z_flat = z.reshape(-1, self.codebook_dim)

        # Calculate distances to codebook vectors
        d = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        # Get closest codebook indices
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # Compute loss
        loss = torch.mean((z_q.detach() - z) ** 2) + self.commitment_cost * torch.mean(
            (z_q - z.detach()) ** 2
        )

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, loss, min_encoding_indices
