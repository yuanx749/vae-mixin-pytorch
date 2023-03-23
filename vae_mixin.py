"""Mixin classes for VAE varients."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VAEMixin:
    """Mixin class for basic VAE.

    https://arxiv.org/abs/1312.6114

    """

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, x: Tensor) -> Tensor:
        """

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        z : torch.Tensor
            The latent variables if training else mean.

        """
        self.mu_, self.logvar_ = self.encoder(x)
        if self.training:
            z = self.reparameterize(self.mu_, self.logvar_)
            return z
        return self.mu_

    def decode(self, z: Tensor) -> Tensor:
        """

        Parameters
        ----------
        z : torch.Tensor
            The latent variables.

        Returns
        -------
        x_rec : torch.Tensor
            The reconstructed data.

        """
        return self.decoder(z)

    def rec_loss(self, x_rec: Tensor, x: Tensor, **kwargs) -> Tensor:
        """Compute the reconstruction loss, when the output :math:`p(x|z)` is Bernoulli.

        Parameters
        ----------
        x_rec : torch.Tensor
            The reconstructed data.
        x : torch.Tensor
            The input data.

        Returns
        -------
        bce : torch.Tensor
            The binary cross entropy.

        """
        bce = F.binary_cross_entropy(x_rec, x, reduction="none").sum(dim=1).mean()
        return bce

    def kl_loss(
        self,
        mu: Optional[Tensor] = None,
        logvar: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Compute the KL divergence loss, when the prior :math:`p(z)=\mathcal{N}(0,\mathbf{I})` and the posterior approximation :math:`q(z|x)` are Gaussian.

        Parameters
        ----------
        mu : torch.Tensor, optional
            Encoder output :math:`\mu`.
        logvar : torch.Tensor, optional
            Encoder output :math:`\log\sigma^2`.

        Returns
        -------
        kld : torch.Tensor
            The KL divergence.

        """
        mu = self.mu_ if mu is None else mu
        logvar = self.logvar_ if logvar is None else logvar
        var = torch.exp(logvar)
        kld = -0.5 * (1 + logvar - mu**2 - var).sum(dim=1).mean()
        return kld

    def loss(self, **kwargs) -> Tensor:
        """Compute the VAE objective function.

        Parameters
        ----------
        **kwargs
            Extra arguments to loss terms.

        Returns
        -------
        torch.Tensor

        """
        return self.rec_loss(**kwargs) + self.kl_loss(**kwargs)


class BetaVAEMixin:
    r"""Mixin class for :math:`\beta`-VAE.

    https://openreview.net/forum?id=Sy2fzU9gl

    Parameters
    ----------
    beta : float, default 1

    """

    def __init__(self, beta: float = 1, **kwargs) -> None:
        self.beta = beta
        super().__init__(**kwargs)

    def loss(self, beta: Optional[float] = None, **kwargs) -> Tensor:
        r"""Compute the :math:`\beta`-VAE objective function.

        Parameters
        ----------
        beta : float, optional
            Coefficient :math:`\beta`.
        **kwargs
            Extra arguments to loss terms.

        Returns
        -------
        torch.Tensor

        """
        beta = self.beta if beta is None else beta
        return self.rec_loss(**kwargs) + beta * self.kl_loss(**kwargs)


class InfoVAEMixin:
    """Mixin class for InfoVAE.

    https://arxiv.org/abs/1706.02262

    Parameters
    ----------
    lamb : float, default 1
    alpha : float, default 0

    """

    def __init__(self, lamb: float = 1, alpha: float = 0, **kwargs) -> None:
        self.lamb = lamb
        self.alpha = alpha
        super().__init__(**kwargs)

    @staticmethod
    def kernel(x1: Tensor, x2: Tensor) -> Tensor:
        """Compute the RBF kernel."""
        n1 = x1.size(0)
        n2 = x2.size(0)
        dim = x1.size(1)
        x1 = x1.view(n1, 1, dim)
        x2 = x2.view(1, n2, dim)
        return torch.exp(-1.0 / dim * (x1 - x2).pow(2).sum(dim=2))

    def mmd_loss(self, z: Tensor, **kwargs) -> Tensor:
        """Compute the MMD loss of :math:`q(z)` and :math:`p(z)`.

        Parameters
        ----------
        z : torch.Tensor
            The latent variables.

        Returns
        -------
        mmd : torch.Tensor
            The squared maximum mean discrepancy.

        """
        z_p = torch.randn_like(z)
        mmd = (
            self.kernel(z_p, z_p).mean()
            - 2 * self.kernel(z, z_p).mean()
            + self.kernel(z, z).mean()
        )
        return mmd

    def loss(
        self,
        lamb: Optional[float] = None,
        alpha: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        r"""Compute the InfoVAE objective function.

        Parameters
        ----------
        lamb : float, optional
            Scaling coefficient :math:`\lambda`.
        alpha : float, optional
            Information preference :math:`\alpha`.
        **kwargs
            Extra arguments to loss terms.

        Returns
        -------
        torch.Tensor

        """
        lamb = self.lamb if lamb is None else lamb
        alpha = self.alpha if alpha is None else alpha
        return (
            self.rec_loss(**kwargs)
            + (1 - alpha) * self.kl_loss(**kwargs)
            + (alpha + lamb - 1) * self.mmd_loss(**kwargs)
        )


class VQVAEMixin:
    """Mixin class for VQ-VAE.

    https://arxiv.org/abs/1711.00937

    Parameters
    ----------
    embedding_dim : int
        The dimensionality of latent embedding vector.
    num_embeddings : int
        The size of the discrete latent space.
    commitment_cost : float

    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        commitment_cost: float,
        **kwargs,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        super().__init__(**kwargs)
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def encode(self, x: Tensor) -> Tensor:
        """

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        z_e : torch.Tensor
            The encoder outputs.

        """
        z_e = self.encoder(x)
        return z_e

    def quantize(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize the inputs vectors.

        Parameters
        ----------
        inputs : torch.Tensor
            The encoder outputs.

        Returns
        -------
        z_q : torch.Tensor
            The quantized vectors of the inputs.
        z : torch.Tensor
            The discrete latent variables.

        """
        assert inputs.size(-1) == self.embedding_dim
        flat_inputs = inputs.reshape(-1, self.embedding_dim)
        distances = (
            (flat_inputs**2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embeddings.weight.t()
            + (self.embeddings.weight**2).sum(dim=1)
        )
        encoding_indices = distances.argmin(dim=1)
        encoding_indices = encoding_indices.reshape(inputs.shape[:-1])
        quantized = self.embeddings.forward(encoding_indices)
        self.codebook_loss_, self.commitment_loss_ = self.vq_loss(inputs, quantized)
        # straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        return quantized, encoding_indices

    def vq_loss(self, z_e: Tensor, z_q: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        """Compute the vector quantization loss terms.

        Parameters
        ----------
        z_e : torch.Tensor
            The encoder outputs.
        z_q : torch.Tensor
            The quantized vectors.

        Returns
        -------
        codebook_loss : torch.Tensor
        commitment_loss : torch.Tensor

        """
        codebook_loss = (z_e.detach() - z_q).pow(2).sum(dim=-1).mean()
        commitment_loss = (z_e - z_q.detach()).pow(2).sum(dim=-1).mean()
        return codebook_loss, commitment_loss

    def loss(self, beta: Optional[float] = None, **kwargs) -> Tensor:
        """Compute the VQ-VAE objective function.

        Parameters
        ----------
        beta : float, optional
            Scalar which controls the weighting of the commitment loss.
        **kwargs
            Extra arguments to loss terms.

        Returns
        -------
        torch.Tensor

        """
        beta = self.commitment_cost if beta is None else beta
        return (
            self.rec_loss(**kwargs) + self.codebook_loss_ + beta * self.commitment_loss_
        )
