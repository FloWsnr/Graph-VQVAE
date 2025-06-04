import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from typing import Literal

from g_vqvae.model.graph.graph_attention import GraphAttentionLayer


class GraphDecoder(nn.Module):
    """
    Graph decoder for the G_VQVAE.
    The decoder only gets the node features and must infer the edges and their features.
    """

    def __init__(
        self,
        codebook_dim: int,
        hidden_dim: int = 256,
        out_channels: int = 1,
        layer_type: Literal["gcn", "gat", "attention"] = "gcn",
    ):
        super().__init__()

        if layer_type == "gcn":
            self.conv1 = gnn.GCNConv(codebook_dim, hidden_dim)
            self.conv2 = gnn.GCNConv(hidden_dim, out_channels)
            self.relu = nn.ReLU()
        elif layer_type == "gat":
            self.conv1 = gnn.GATConv(codebook_dim, hidden_dim)
            self.conv2 = gnn.GATConv(hidden_dim, out_channels)
            self.relu = nn.ReLU()
        elif layer_type == "attention":
            self.conv1 = GraphAttentionLayer(codebook_dim, hidden_dim)
            self.conv2 = GraphAttentionLayer(hidden_dim, out_channels)
            self.relu = nn.ReLU()
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")

    def forward(self, x: torch.Tensor, num_nodes: int) -> Data:
        """
        Forward pass through the decoder.
        """
        # add nodes so we get original number of nodes
        missing_nodes = num_nodes - x.shape[0]
        x = torch.cat([x, torch.zeros(missing_nodes, x.shape[1])], dim=0)

        # Create a fully connected edge index for the graph using vectorization
        i = torch.arange(num_nodes, device=x.device)
        j = torch.arange(num_nodes, device=x.device)
        ii, jj = torch.meshgrid(i, j, indexing="ij")
        mask = ii != jj  # Skip self-loops
        edge_index = torch.stack([ii[mask], jj[mask]], dim=0)

        # add edge attributes to the graph, features are zero for new edges
        edge_attr = torch.zeros(edge_index.shape[1], 1)

        # First convolution
        if hasattr(self.conv1, 'forward') and isinstance(self.conv1, GraphAttentionLayer):
            temp_data = Data(x=x, edge_index=edge_index)
            temp_data = self.conv1(temp_data)
            x = temp_data.x
        else:
            x = self.conv1(x, edge_index)
        x = self.relu(x)
        
        # Second convolution
        if hasattr(self.conv2, 'forward') and isinstance(self.conv2, GraphAttentionLayer):
            temp_data = Data(x=x, edge_index=edge_index)
            temp_data = self.conv2(temp_data)
            x = temp_data.x
        else:
            x = self.conv2(x, edge_index)
        x = self.relu(x)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


if __name__ == "__main__":
    # Test the decoder
    from torch_geometric.datasets import FakeDataset

    dataset = FakeDataset(
        num_graphs=1,
        avg_num_nodes=10,
        avg_degree=4,
        num_node_features=32,
        num_edge_features=8,
    )
    data = dataset[0]
