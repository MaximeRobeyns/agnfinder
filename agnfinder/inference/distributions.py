# AGNfinder: Detect AGN from photometry in XXL data.
#
# Copyright (C) 2021 Maxime Robeyns <maximerobeyns@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
"""Common distributions for use in CVAE."""

import math
import torch as t
import torch.distributions as dist
import torch.distributions.utils as distutils

from agnfinder.types import Tensor
from agnfinder.inference.base import _CVAE_Dist, _CVAE_RDist


# Reparametrised distributions ================================================


class R_Gaussian(_CVAE_RDist, dist.Normal):
    """Reparametrised factorised Gaussian distribution

    Here we simply make use of PyTorch's existing Normal distribution, which
    implements the abstract methods required by _CVAE_RDist: log_prob and
    rsample.
    """

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        _CVAE_RDist.__init__(self)
        dist.Normal.__init__(self, mean, std)
        assert self.has_rsample

    def log_prob(self, value: Tensor, nojoint: bool = False) -> Tensor:
        lp = dist.Normal.log_prob(self, value)
        return lp if nojoint else lp.sum(1)

    def rsample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        return dist.Normal.rsample(self, sample_shape)


class Manual_R_Gaussian(_CVAE_RDist):
    """
    Home baked reparametrised Gaussian distribution---this should be
    identical to the PyTorch wrapper above.
    """

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        self.mean, self.std = distutils.broadcast_all(mean, std)
        super().__init__(batch_shape=self.mean.size(), device=self.mean.device,
                         dtype=self.mean.dtype)
        assert self.mean.device == self.std.device
        assert self.mean.dtype == self.std.dtype

    def log_prob(self, _: Tensor, nojoint: bool = False) -> Tensor:
        log2pi = math.log(2 * math.pi)
        nlp = 0.5 * (log2pi + t.pow(self.last_eps, 2.)) + t.log(self.std)
        return -nlp if nojoint else -t.sum(nlp, 1)

    def rsample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        # create eps on same device / dtype as mean parameter.
        self.last_eps = t.randn(shape, device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + self.last_eps * self.std


class R_MVN(_CVAE_RDist):
    """
    Reparametrised multivariate normal distribution (implementing only
    essential methods)
    """
    def __init__(self, mean: Tensor, L: Tensor) -> None:
        """
        Reparametrised multivariate normal (Gaussian) distribution.

        Args:
            mean: the mean vector
            L: a lower-triangular matrix, such that LL^{T} = Sigma.
        """
        super().__init__(batch_shape=mean.size(), device=mean.device,
                         dtype=mean.dtype)
        assert mean.device == L.device
        assert mean.dtype == L.dtype
        # TODO implement broadcast_all
        # old_L_shape = L.shape
        # self.mean, self.L = distutils.broadcast_all(mean, L.flatten(-2))
        # self.L = self.L.view(old_L_shape)
        self.mean = mean
        self.std = L.sum(-1)
        self.cov = t.bmm(L, L.transpose(-2, -1))

    def log_prob(self, _:Tensor, nojoint: bool = False) -> Tensor:
        log2pi = math.log(2 * math.pi)
        nlp = 0.5 * (log2pi + t.pow(self.last_eps, 2.)) + t.log(self.std)
        return -nlp if nojoint else -t.sum(nlp, 1)

    def rsample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        self.last_eps = t.randn(shape, device=self.mean.device, dtype=self.mean.dtype)
        cov = self.cov @ self.last_eps.unsqueeze(-1)
        return self.mean + cov.squeeze()


# Simple distributions ========================================================


class Gaussian(_CVAE_Dist, dist.Normal):
    """Factorised Gaussian distribution (PyTorch wrapper)"""

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        _CVAE_Dist.__init__(self)
        dist.Normal.__init__(self, mean, std)

    def log_prob(self, value: Tensor, nojoint: bool = False) -> Tensor:
        # product of univariate Gaussian densities
        lp = dist.Normal.log_prob(self, value)
        return lp if nojoint else lp.sum(1)

    def sample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        return dist.Normal.sample(self, sample_shape)


class Laplace(_CVAE_Dist, dist.Laplace):
    def __init__(self, loc: Tensor, scale: Tensor) -> None:
        _CVAE_Dist.__init__(self)
        dist.Laplace.__init__(self, loc, scale)

    def log_prob(self, value: Tensor, nojoint: bool = False) -> Tensor:
        lp = dist.Laplace.log_prob(self, value)
        return lp if nojoint else lp.sum(1)

    def sample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        return dist.Laplace.sample(self, sample_shape)


class Manual_Gaussian(_CVAE_Dist):
    """
    Roll-your-own implementation of a factorised Gaussian distribution---once
    again, this should be identical to the PyTorch wrapper above.
    """

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        self.mean, self.std = distutils.broadcast_all(mean, std)
        _CVAE_Dist.__init__(self, batch_shape=self.mean.size(), device=self.mean.device,
                            dtype=self.mean.dtype)
        assert self.mean.device == self.std.device
        assert self.mean.dtype == self.std.dtype

    def log_prob(self, value: Tensor, nojoint: bool = False) -> Tensor:
        var = self.std ** 2
        nlp = 0.5 * (t.log(2. * math.pi * var) +
                        (value - self.mean)**2
                     / var)
        return -nlp if nojoint else -t.sum(nlp, 1)

    def sample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        with t.no_grad():
            return t.normal(self.mean.expand(shape), self.std.expand(shape))


class MVN(_CVAE_Dist, dist.MultivariateNormal):
    """Multivariate Gaussian

    Warning: very slow / expensive.
    """
    def __init__(self, mean: Tensor, L: Tensor) -> None:
        _CVAE_Dist.__init__(self)
        dist.MultivariateNormal.__init__(self, mean, scale_tril=L)

    def log_prob(self, value: Tensor, _: bool = False) -> Tensor:
        return dist.MultivariateNormal.log_prob(self, value)

    def sample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        return dist.MultivariateNormal.sample(self, sample_shape)


class Multinomial(_CVAE_Dist, dist.Multinomial):
    """Multinomial distribution (PyTorch wrapper)"""

    def __init__(self, params: Tensor) -> None:
        _CVAE_Dist.__init__(self)
        dist.Multinomial.__init__(self, 1, params)

    def log_prob(self, value: Tensor, _: bool = False) -> Tensor:
        return dist.Multinomial.log_prob(self, value)

    def sample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        return dist.Multinomial.sample(self, sample_shape)
