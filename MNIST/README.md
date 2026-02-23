# VAE on MNIST

A Variational Autoencoder trained on the MNIST handwritten digits dataset using PyTorch.

## Architecture

A simple linear VAE with a two-layer encoder and decoder.

```
Input (784)  →  Hidden (200)  →  μ, σ (20)  →  Hidden (200)  →  Output (784)
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input Dim | 784 |
| Hidden Dim | 200 |
| Latent Dim | 20 |
| Batch Size | 128 |
| Learning Rate | 3e-4 |
| Epochs | 10 |
| Optimizer | Adam |

## Usage

**Train:**
```bash
python train.py
```

**Generated outputs:**
- `generated_{digit}_ex{n}.png` — generated examples for each digit 0-9

## Results

<!-- Add your inference results here -->

---

*For a full explanation of how VAEs work see [VAE.md](VAE.md)*
