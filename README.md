# physics-vqvae

Using Vector-Quantized VAEs to model physics

## Installation

```bash
conda create -n g_vqvae python=3.12
conda activate g_vqvae
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install einops h5py imageio ipykernel matplotlib pandas wandb dotenv prodigyopt
pip install scipy pytest
pip install -e .
```

## Usage

To train the VQVAE on a dataset, run the training script.
Before this, make sure you have a dataset in correct folder.
Additionally, wandb requires an api key to be set, we assume it is placed in a .env file in the root directory.
```
WANDB_API_KEY=xxx
```

