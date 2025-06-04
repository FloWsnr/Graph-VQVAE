# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Environment Setup
```bash
conda create -n g_vqvae python=3.12
conda activate g_vqvae
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install einops h5py imageio ipykernel matplotlib pandas wandb dotenv prodigyopt
pip install scipy pytest
pip install -e .
```

### Testing
```bash
pytest                    # Run all tests
pytest tests/test_model/   # Run model tests only
pytest -v                 # Verbose output
```

### Environment Variables
The project requires a `.env` file in the root directory with:
```
WANDB_API_KEY=xxx
```

## Architecture

This is a Graph Vector-Quantized VAE (VQ-VAE) implementation for modeling physics using graph neural networks.

### Core Components

1. **Graph VQ-VAE Model** (`g_vqvae/model/graph/graph_vqvae.py`):
   - `G_VQVAE`: Main model combining GNN encoder/decoder with vector quantization
   - `GNN`: Graph neural network using GCNConv layers from PyTorch Geometric
   - Encodes graphs → quantizes node features → decodes back to graphs

2. **Vector Quantization** (`g_vqvae/model/graph/graph_vq.py`):
   - `GraphVectorQuantizer`: Quantizes node features while preserving edge structure
   - Works with `torch_geometric.data.Data` objects
   - Maintains edge_index and edge_attr during quantization

3. **Data Handling** (`g_vqvae/data/dataset.py`):
   - Supports standard datasets: QM7b, QM9, ZINC
   - Uses PyTorch Geometric dataset loaders

4. **Image VQ-VAE** (`g_vqvae/model/image/`):
   - Traditional VQ-VAE implementation for comparison/baseline
   - `VQVAE` and `VectorQuantizer` classes

### Key Design Patterns

- All models inherit from `torch.nn.Module`
- Graph data uses `torch_geometric.data.Data` format
- VQ modules return (quantized_data, loss, indices) tuples
- Encode/decode methods are separated for flexibility
- Test fixtures use FakeDataset for consistent testing

### Dependencies

- PyTorch + PyTorch Geometric for graph neural networks
- Standard ML libraries: einops, wandb, matplotlib
- Testing with pytest