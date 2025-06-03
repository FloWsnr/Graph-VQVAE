import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data

from g_vqvae.model.graph_vq import GraphVectorQuantizer


class GNN(nn.Module):
    """
    Graph Neural Network encoder/decoder.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 256):
        super().__init__()

        self.conv1 = gnn.GCNConv(in_channels, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, data: Data) -> Data:
        """
        Forward pass through the GNN.
        """
        x = data.x
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)

        return Data(x=x, edge_index=edge_index)


class G_VQVAE(nn.Module):
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

        # Encoder and decoder GNNs
        self.encoder = GNN(in_channels, hidden_dim)
        self.decoder = GNN(codebook_dim, hidden_dim)

        # Projection layers for codebook
        self.to_codebook = nn.Linear(hidden_dim, codebook_dim)
        self.from_codebook = nn.Linear(hidden_dim, in_channels)

        # Vector Quantizer
        self.quantizer = GraphVectorQuantizer(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            commitment_cost=commitment_cost,
        )

    def encode(self, data: Data) -> tuple[Data, torch.Tensor, torch.Tensor]:
        """
        Encode the input graph.

        Parameters
        ----------
        data : Data
            Input graph data containing node features and edge_index

        Returns
        -------
        tuple[Data, torch.Tensor, torch.Tensor]
            - Quantized graph data
            - Codebook loss
            - Encoding indices
        """
        # Encode through GNN
        encoded_data: Data = self.encoder(data)

        # Project to codebook dimension
        z = self.to_codebook(encoded_data.x)
        quant_data = Data(
            x=z,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
        )

        # Quantize
        quantized_data, loss, indices = self.quantizer(quant_data)

        return quantized_data, loss, indices

    def decode(self, data: Data) -> Data:
        """
        Decode the quantized graph representation.

        Parameters
        ----------
        data : Data
            Quantized graph data containing node features and edge_index

        Returns
        -------
        Data
            Reconstructed graph data
        """
        # Decode through GNN
        decoded_data = self.decoder(data)

        # Project back to input dimension
        x_recon = self.from_codebook(decoded_data.x)

        return Data(
            x=x_recon,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
        )

    def forward(self, data: Data) -> tuple[Data, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VQ-GNN.

        Parameters
        ----------
        data : Data
            Input graph data containing node features and edge_index

        Returns
        -------
        tuple[Data, torch.Tensor, torch.Tensor]
            - Reconstructed graph data
            - Codebook loss
            - Encoding indices
        """
        # Encode
        quantized_data, loss, indices = self.encode(data)
        # Decode
        reconstructed_data = self.decode(quantized_data)

        return reconstructed_data, loss, indices
