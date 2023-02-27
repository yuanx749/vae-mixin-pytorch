# vae-mixin-pytorch

Variational autoencoders as mixins.

This repo contains implementation of variational autoencoder (VAE) and variants in PyTorch as mixin classes, which can be reused and composed in your customized modules.

## Usage

An example using simple encoder and decoder on the MNIST dataset is in `example.py`.

## Notes

Implemented VAEs:
- VAE
- beta-VAE
- InfoVAE
- VQ-VAE

Losses are averaged across samples, and summed along each latent vector in a minibatch.
