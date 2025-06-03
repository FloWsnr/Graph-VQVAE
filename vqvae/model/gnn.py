import torch_geometric.nn as nn

class GNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = nn.GCNConv(in_channels, out_channels)
        self.conv2 = nn.GCNConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x