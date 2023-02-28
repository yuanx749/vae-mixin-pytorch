# vae-mixin-pytorch

Variational autoencoders as mixins.

This repo contains implementation of variational autoencoder (VAE) and variants in PyTorch as mixin classes, which can be reused and composed in your customized modules.

## Usage

Check the docs [here](https://yuanx749.github.io/vae-mixin-pytorch/).

An example using simple encoder and decoder on the MNIST dataset is in `example.py`.

```{note}
Mixin is a term in object-oriented programming.
```

## Notes

Implemented VAEs:
- VAE
- beta-VAE
- InfoVAE
- VQ-VAE

```{note}
Losses are averaged across samples, and summed along each latent vector in a minibatch.
```
