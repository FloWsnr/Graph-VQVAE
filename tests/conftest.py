"""
Fixtures for testing

By: Florian Wiesner
Date: 2025-05-12
"""

import pytest

from torch_geometric.datasets import FakeDataset


@pytest.fixture
def torch_geometric_graph():
    dataset = FakeDataset(num_graphs=1, avg_num_nodes=10, avg_degree=4)
    return dataset[0]
