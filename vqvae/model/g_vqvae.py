import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from vqvae.model.vq import VectorQuantizer


class GNN(nn.Module):
    """
    Graph Neural Network encoder/decoder with vector quantization.

    Parameters
    ----------
    in_channels : int
        Number of input features per node
    hidden_dim : int
        Hidden dimension for the GNN layers
    codebook_size : int
        Number of embeddings in the codebook
    codebook_dim : int
        Dimension of each embedding vector
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

        # Encoder
        self.encoder = nn.Sequential(
            gnn.GCNConv(in_channels, hidden_dim),
            nn.ReLU(),
            gnn.GCNConv(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Projection layers for codebook
        self.to_codebook = nn.Linear(hidden_dim, codebook_dim)
        self.from_codebook = nn.Linear(codebook_dim, hidden_dim)

        # Vector Quantizer
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            commitment_cost=commitment_cost,
        )

        # Decoder
        self.decoder = nn.Sequential(
            gnn.GCNConv(hidden_dim, hidden_dim),
            nn.ReLU(),
            gnn.GCNConv(hidden_dim, in_channels),
        )

    def encode(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode the input graph.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (num_nodes, in_channels)
        edge_index : torch.Tensor
            Graph connectivity of shape (2, num_edges)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Quantized tensor
            - Codebook loss
            - Encoding indices
        """
        z = self.encoder(x, edge_index)
        z = self.to_codebook(z)
        z_q, loss, indices = self.quantizer(z)
        return z_q, loss, indices

    def decode(self, z_q: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Decode the quantized graph representation.

        Parameters
        ----------
        z_q : torch.Tensor
            Quantized tensor of shape (num_nodes, codebook_dim)
        edge_index : torch.Tensor
            Graph connectivity of shape (2, num_edges)

        Returns
        -------
        torch.Tensor
            Reconstructed node features
        """
        z_q = self.from_codebook(z_q)
        x_recon = self.decoder(z_q, edge_index)
        return x_recon

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VQ-GNN.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (num_nodes, in_channels)
        edge_index : torch.Tensor
            Graph connectivity of shape (2, num_edges)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Reconstructed node features
            - Codebook loss
            - Encoding indices
        """
        # Encode
        z_q, loss, indices = self.encode(x, edge_index)
        # Decode
        x_recon = self.decode(z_q, edge_index)

        return x_recon, loss, indices
