"""Mixin classes for VAE varients."""

import math
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
        eps = torch.randn_like(std)
        return mu + std * eps

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
        mu, logvar = self.encoder(x)
        self.mu_, self.logvar_ = mu, logvar
        if self.training:
            z = self.reparameterize(mu, logvar)
            return z
        return mu

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
        self.loss_rec_ = bce
        return bce

    def kl_loss(
        self,
        mu: Optional[Tensor] = None,
        logvar: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Compute the KL divergence loss, when the prior :math:`p(z)=\mathcal{N}(0,\mathbf{I})`
        and the posterior approximation :math:`q(z|x)` are Gaussian.

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
        self.loss_kl_ = kld
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
        Regularisation coefficient :math:`\beta`.

    """

    def __init__(self, beta: float = 1, **kwargs) -> None:
        self.beta = beta
        super().__init__(**kwargs)

    def loss(self, **kwargs) -> Tensor:
        r"""Compute the :math:`\beta`-VAE objective function.

        Parameters
        ----------
        **kwargs
            Extra arguments to loss terms.

        Returns
        -------
        torch.Tensor

        """
        return self.rec_loss(**kwargs) + self.beta * self.kl_loss(**kwargs)


class InfoVAEMixin:
    r"""Mixin class for InfoVAE.

    https://arxiv.org/abs/1706.02262

    Parameters
    ----------
    lamb : float, default 1
        Scaling coefficient :math:`\lambda`.
    alpha : float, default 0
        Information preference :math:`\alpha`.

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
        qz = z
        pz = torch.randn_like(z)
        mmd = (
            self.kernel(pz, pz).mean()
            - 2 * self.kernel(qz, pz).mean()
            + self.kernel(qz, qz).mean()
        )
        self.loss_mmd_ = mmd
        return mmd

    def loss(self, **kwargs) -> Tensor:
        """Compute the InfoVAE objective function.

        Parameters
        ----------
        **kwargs
            Extra arguments to loss terms.

        Returns
        -------
        torch.Tensor

        """
        return (
            self.rec_loss(**kwargs)
            + (1 - self.alpha) * self.kl_loss(**kwargs)
            + (self.alpha + self.lamb - 1) * self.mmd_loss(**kwargs)
        )


class DIPVAEMixin:
    """Mixin class for DIP-VAE.

    https://arxiv.org/abs/1711.00848

    Parameters
    ----------
    lambda_od : float, default 10
        Hyperparameter :math:`\lambda_{od}` for the loss on the off-diagonal entries.
    lambda_d : float, default 10
        Hyperparameter :math:`\lambda_d` for the loss on the diagonal entries.
    dip : int, 1 or 2
        DIP type.

    """

    def __init__(
        self,
        lambda_od: float = 10,
        lambda_d: float = 10,
        dip: int = 1,
        **kwargs,
    ) -> None:
        self.lambda_od = lambda_od
        self.lambda_d = lambda_d
        self.dip = dip
        super().__init__(**kwargs)

    def dip_regularizer(
        self,
        mu: Optional[Tensor] = None,
        logvar: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the DIP regularization terms, when :math:`q(z|x)` is Gaussian.

        Parameters
        ----------
        mu : torch.Tensor, optional
            Encoder output :math:`\mu`.
        logvar : torch.Tensor, optional
            Encoder output :math:`\log\sigma^2`.

        Returns
        -------
        off_diag_loss : torch.Tensor
            The off-diagonal loss.
        diag_loss : torch.Tensor
            The diagonal loss.

        """
        mu = self.mu_ if mu is None else mu
        logvar = self.logvar_ if logvar is None else logvar
        mu_mean = mu - mu.mean(dim=0)
        cov_mu = mu_mean.t() @ mu_mean / mu.size(0)
        if self.dip == 1:
            cov = cov_mu
        elif self.dip == 2:
            var = torch.diag_embed(torch.exp(logvar))
            cov = var.mean(dim=0) + cov_mu
        else:
            raise ValueError
        cov_diag = torch.diag(cov)
        cov_off_diag = cov - torch.diag(cov_diag)
        off_diag_loss = cov_off_diag.pow(2).sum()
        diag_loss = (cov_diag - 1).pow(2).sum()
        self.loss_off_diag_, self.loss_diag_ = off_diag_loss, diag_loss
        return off_diag_loss, diag_loss

    def loss(self, **kwargs) -> Tensor:
        """Compute the DIP-VAE objective function.

        Parameters
        ----------
        **kwargs
            Extra arguments to loss terms.

        Returns
        -------
        torch.Tensor

        """
        off_diag_loss, diag_loss = self.dip_regularizer(**kwargs)
        return (
            self.rec_loss(**kwargs)
            + self.kl_loss(**kwargs)
            + self.lambda_od * off_diag_loss
            + self.lambda_d * diag_loss
        )


class BetaTCVAEMixin:
    r"""Mixin class for :math:`\beta`-TCVAE.

    https://arxiv.org/abs/1802.04942

    Parameters
    ----------
    alpha : float, default 1
        Weight :math:`\alpha` of the index-code mutual information.
    beta : float, default 1
        Weight :math:`\beta` of the total correlation.
    gamma : float, default 1
        Weight :math:`\gamma` of the dimension-wise KL.

    """

    def __init__(
        self,
        alpha: float = 1,
        beta: float = 1,
        gamma: float = 1,
        **kwargs,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        super().__init__(**kwargs)

    @staticmethod
    def gaussian_log_density(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        """Compute the log density of a Gaussian."""
        log2pi = math.log(2 * math.pi)
        inv_var = torch.exp(-logvar)
        return -0.5 * (log2pi + logvar + (x - mu).pow(2) * inv_var)

    def decompose_kl(
        self,
        n: int,
        z: Tensor,
        mu: Optional[Tensor] = None,
        logvar: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the decomposed KL terms.

        Parameters
        ----------
        n : int
            Dataset size :math:`N`.
        z : torch.Tensor
            The latent variables.
        mu : torch.Tensor, optional
            Encoder output :math:`\mu`.
        logvar : torch.Tensor, optional
            Encoder output :math:`\log\sigma^2`.

        Returns
        -------
        mi_loss : torch.Tensor
            The index-code mutual information.
        tc_loss : torch.Tensor
            The total correlation.
        kl_loss : torch.Tensor
            The dimension-wise KL.

        """
        mu = self.mu_ if mu is None else mu
        logvar = self.logvar_ if logvar is None else logvar
        m, dim = z.shape
        logqz_x = self.gaussian_log_density(z, mu, logvar).sum(dim=1)
        zeros = torch.zeros_like(z)
        logpz = self.gaussian_log_density(z, zeros, zeros).sum(dim=1)
        # minibatch weighted sampling
        logqzxi_xj = self.gaussian_log_density(
            z.view(m, 1, dim), mu.view(1, m, dim), logvar.view(1, m, dim)
        )
        logqz = torch.logsumexp(logqzxi_xj.sum(dim=2), dim=1) - math.log(n * m)
        logprodqz = (torch.logsumexp(logqzxi_xj, dim=1) - math.log(n * m)).sum(dim=1)
        mi_loss = (logqz_x - logqz).mean()
        tc_loss = (logqz - logprodqz).mean()
        kl_loss = (logprodqz - logpz).mean()
        self.loss_mi_, self.loss_tc_, self.loss_kl_ = mi_loss, tc_loss, kl_loss
        return mi_loss, tc_loss, kl_loss

    def loss(self, **kwargs) -> Tensor:
        r"""Compute the :math:`\beta`-TCVAE objective function.

        Parameters
        ----------
        **kwargs
            Extra arguments to loss terms.

        Returns
        -------
        torch.Tensor

        """
        mi_loss, tc_loss, kl_loss = self.decompose_kl(**kwargs)
        return (
            self.rec_loss(**kwargs)
            + self.alpha * mi_loss
            + self.beta * tc_loss
            + self.gamma * kl_loss
        )


class VQVAEMixin:
    r"""Mixin class for VQ-VAE.

    https://arxiv.org/abs/1711.00937

    Parameters
    ----------
    embedding_dim : int
        The dimensionality of latent embedding vector.
    num_embeddings : int
        The size of the discrete latent space.
    commitment_cost : float
        Scalar :math:`\beta` which controls the weighting of the commitment loss.

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
        self.loss_codebook_, self.loss_commitment_ = self.vq_loss(inputs, quantized)
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

    def loss(self, **kwargs) -> Tensor:
        """Compute the VQ-VAE objective function.

        Parameters
        ----------
        **kwargs
            Extra arguments to loss terms.

        Returns
        -------
        torch.Tensor

        """
        beta = self.commitment_cost
        return (
            self.rec_loss(**kwargs) + self.loss_codebook_ + beta * self.loss_commitment_
        )
