import torch
import torch.nn as nn
from torch_geometric.data import Data


class GraphVectorQuantizer(nn.Module):
    """
    Vector Quantizer module specifically designed for graph data that preserves edge structure.

    This quantizer only quantizes node features while keeping the edge structure intact.
    It works with torch_geometric.data.Data objects and preserves their edge_index.

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

    def forward(self, data: Data) -> tuple[Data, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the vector quantizer for graph data.

        Parameters
        ----------
        data : Data
            Input graph data containing node features and edge_index

        Returns
        -------
        tuple[Data, torch.Tensor, torch.Tensor]
            - Quantized graph data with same edge structure
            - Codebook loss
            - Encoding indices for node features
        """
        # Get node features
        z = data.x

        # Calculate distances to codebook vectors
        d = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z, self.embedding.weight.t())
        )

        # Get closest codebook indices
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices)

        # Compute loss
        loss = torch.mean((z_q.detach() - z) ** 2) + self.commitment_cost * torch.mean(
            (z_q - z.detach()) ** 2
        )

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Create new Data object with quantized features but same edge structure
        quantized_data = Data(
            x=z_q,
            edge_index=data.edge_index,
            **{k: v for k, v in data.items() if k not in ["x", "edge_index"]},
        )

        return quantized_data, loss, min_encoding_indices
