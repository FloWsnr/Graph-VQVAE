import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Dict, Any, Optional

from g_vqvae.model.graph.graph_vq import GraphVectorQuantizer
from g_vqvae.model.graph.encoder import GraphEncoder
from g_vqvae.model.graph.decoder import GraphDecoder

class G_VQVAE(nn.Module):
    """
    Graph Neural Network Autoencoder with vector quantization.

    The model convert the input graph to a fully connected graph,
    performs feature extraction, quantizes the features, and reconstructs the graph.

    Currently, no edge features are used.

    Parameters
    ----------
    in_channels : int
        Number of input features per node
    hidden_dim : int
        Hidden dimension for the encoder/decoder layers
    codebook_size : int
        Number of embeddings in the codebook
    codebook_dim : int
        Dimension of each embedding vector
    commitment_cost : float
        Commitment cost for the codebook loss
    encoder_config : dict, optional
        Configuration for the encoder (layer_type, pooling_type, etc.)
    decoder_config : dict, optional
        Configuration for the decoder (layer_type, unpooling_type, etc.)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        codebook_size: int = 512,
        codebook_dim: int = 256,
        commitment_cost: float = 0.25,
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        # Default configurations
        if encoder_config is None:
            encoder_config = {"layer_type": "gcn", "pooling_type": "SAG"}
        if decoder_config is None:
            decoder_config = {"layer_type": "gcn"}

        # Create encoder
        self.encoder = GraphEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            layer_type=encoder_config["layer_type"],
            pooling_type=encoder_config["pooling_type"],
        )

        # Create decoder
        self.decoder = GraphDecoder(
            codebook_dim=codebook_dim,
            hidden_dim=hidden_dim,
            out_channels=in_channels,
            layer_type=decoder_config["layer_type"],
        )

        # Vector Quantizer
        self.quantizer = GraphVectorQuantizer(
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            commitment_cost=commitment_cost,
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
        # store the original num nodes
        num_nodes = data.num_nodes
        # Encode the graph with pooling (i.e. less nodes)
        z_graph = self.encoder(data)

        # Quantize
        z_graph, loss, indices = self.quantizer(z_graph)

        # Decode the graph with unpooling
        # the decoder only gets the node features and must infer the edges
        data_rec = self.decoder(z_graph.x, num_nodes)

        return data_rec, loss, indices
