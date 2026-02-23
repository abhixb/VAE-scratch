# VAE on CelebA

A Variational Autoencoder trained on the CelebA celebrity faces dataset using PyTorch.

## Why Convolutional?

The first version of this model used a fully connected (linear) VAE — the same architecture as the MNIST model but scaled up to handle `3x64x64` images. It failed. The loss barely moved across 10 epochs because linear layers treat each of the 12,288 pixels as independent values with no relationship to their neighbors.

Switching to a Convolutional VAE fixed this. Convolutional layers understand spatial structure — early layers learn edges and colors, deeper layers combine those into eyes, noses, and face shapes. This hierarchical spatial understanding is what makes the model actually learn faces instead of just averaging them all into one blob.

## Architecture

### Encoder
4 strided `Conv2d` layers progressively downsample the image from `64x64` to a `4x4` feature map, which is then flattened and projected to `μ` and `logvar`.

```
(3, 64, 64) → Conv2d → (32, 32, 32) → Conv2d → (64, 16, 16) → Conv2d → (128, 8, 8) → Conv2d → (256, 4, 4)
                                                                                                      ↓
                                                                                              Linear → μ, logvar
```

### Decoder
A linear layer projects `z` back to `256x4x4`, then 4 `ConvTranspose2d` layers upsample back to the original image size.

```
z (128) → Linear → (256, 4, 4) → ConvTranspose2d → (128, 8, 8) → ConvTranspose2d → (64, 16, 16) → ConvTranspose2d → (32, 32, 32) → ConvTranspose2d → (3, 64, 64)
```

## Numerical Stability Fixes

Two issues came up during training that are worth noting:

**Exploding KL divergence** — the original code learned `σ` directly which could blow up to arbitrarily large values causing `log(σ²)` to go to infinity. Fixed by learning `logvar` instead and working entirely in log space:
```python
logvar = clamp(hid_2logvar(h), -4, 4)
sigma  = exp(0.5 * logvar)
```

**BCELoss out of range** — pixel values of exactly `0.0` or `1.0` cause `log(0)` inside BCELoss. Fixed by clamping both the input images and decoder output to `[1e-6, 1-1e-6]`.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 64 x 64 |
| Latent Dim | 128 |
| Hidden Channels | 32, 64, 128, 256 |
| Batch Size | 128 |
| Learning Rate | 3e-4 |
| Epochs | 40 |
| KL Weight (β) | 0.001 |
| Optimizer | Adam |

The KL weight `β = 0.001` is intentionally low. A full weight of `1.0` caused mode collapse — the model learned one average face for everything. Lowering `β` lets the encoder spread different faces across different regions of the latent space.

## Dataset Setup

CelebA does not reliably download via torchvision due to Google Drive rate limits. Download the images manually from the [official site](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and place them at:

```
dataset/
  celeba/
    img_align_celeba/
      000001.jpg
      000002.jpg
      ...
```

The code uses a custom `Dataset` class that reads directly from this folder, no annotation files needed.

## Usage

**Train:**
```bash
python train.py
```

**Generated outputs:**
- `reconstructions_epoch{n}.png` — real vs reconstructed faces saved after each epoch
- `generated_samples.png` — faces sampled from the latent space after training
- `vae_celeba.pth` — saved model weights

**Load and generate:**
```python
import torch
from model import VariationalAutoEncoder

model = VariationalAutoEncoder(z_dim=128)
model.load_state_dict(torch.load("vae_celeba.pth"))
model.eval()

z = torch.randn(16, 128)
samples = model.decode(z)
```

## Results

<p align="center">
  <img width="1058" height="134" alt="image" src="https://github.com/user-attachments/assets/87a194a3-2611-4876-98f7-cb9cb8a511b4" />
  <br>
   <em>inferenced</em>
</p>
---

*For a full explanation of how VAEs work see [VAE.md](VAE.md)*

