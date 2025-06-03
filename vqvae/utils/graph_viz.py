import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data


def plot_mol(
    data: Data,
    node_features: torch.Tensor,
    edge_features: torch.Tensor,
    ax=None,
):
    """
    Plots a torch_geometric.data.Data object using networkx and matplotlib.
    Optionally displays node and edge features if keys are provided.

    Parameters
    ----------
    data : Data
        PyTorch Geometric Data object with nodes and edges
    node_features : torch.Tensor
        Node features to display as labels
    edge_features : torch.Tensor
        Edge features to display as labels
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis to plot on

    Returns
    -------
    None
    """
    G = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()
    num_nodes = data.num_nodes
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)

    pos = nx.spring_layout(G)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Node labels
    if node_features is not None:
        node_labels = {i: str(node_features[i].cpu().numpy()) for i in range(num_nodes)}
    else:
        node_labels = {i: str(i) for i in range(num_nodes)}

    # Edge labels
    if edge_features is not None:
        edge_labels = {
            (int(u), int(v)): str(edge_features[i].cpu().numpy())
            for i, (u, v) in enumerate(edges)
        }
    else:
        edge_labels = None

    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        labels=node_labels,
        node_color="skyblue",
        edge_color="gray",
        node_size=500,
    )
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax, font_color="red"
        )
    ax.set_title("Graph Visualization")
    plt.tight_layout()
    if ax is None:
        plt.show()
