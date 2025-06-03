from pathlib import Path
from torch_geometric.datasets import QM7b, QM9, ZINC


def get_standard_dataset(name: str, save_dir: Path):
    if name == "qm7b":
        dataset = QM7b(root=save_dir)
    elif name == "qm9":
        dataset = QM9(root=save_dir)
    elif name == "zinc":
        dataset = ZINC(root=save_dir, subset=True)
    else:
        raise ValueError(f"Dataset {name} not found")
    return dataset


def get_custom_dataset(name: str, save_dir: Path):
    if name == "qm7b":
        dataset = QM7b(root=save_dir)