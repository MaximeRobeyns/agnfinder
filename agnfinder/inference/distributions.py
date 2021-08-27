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

    def log_prob(self, value: Tensor) -> Tensor:
        return dist.Normal.log_prob(self, value).sum(1)

    def rsample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        return dist.Normal.rsample(self, sample_shape)


class _Manual_R_Gaussian(_CVAE_RDist):
    """
    Home baked reparametrised Gaussian distribution---this should be
    identical to the PyTorch wrapper above.
    """

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        assert mean.device == std.device
        assert mean.dtype == std.dtype
        self.mean, self.std = distutils.broadcast_all(mean, std)
        super(_CVAE_RDist, self).__init__(batch_shape=self.mean.size())

    def log_prob(self, _: Tensor) -> Tensor:
        log2pi = math.log(2 * math.pi)
        return -t.sum(0.5 * (log2pi + t.pow(self.last_eps, 2.)) + t.log(self.std), 1)

    def rsample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        # create eps on same device / dtype as mean parameter.
        self.last_eps = t.randn(shape, device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + self.last_eps * self.std


# Simple distributions ========================================================


class Gaussian(_CVAE_Dist, dist.Normal):
    """Factorised Gaussian distribution (PyTorch wrapper)"""

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        _CVAE_Dist.__init__(self)
        dist.Normal.__init__(self, mean, std)

    def log_prob(self, value: Tensor) -> Tensor:
        # product of univariate Gaussian densities
        return dist.Normal.log_prob(self, value).sum(1)

    def sample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        return dist.Normal.sample(self, sample_shape)


class Multinomial(_CVAE_Dist, dist.Multinomial):
    """Multinomial distribution (PyTorch wrapper)"""

    def __init__(self, params: Tensor) -> None:
        _CVAE_Dist.__init__(self)
        dist.Multinomial.__init__(self, 1, params)

    def log_prob(self, value: Tensor) -> Tensor:
        return dist.Multinomial.log_prob(self, value)

    def sample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        return dist.Multinomial.sample(self, sample_shape)


class _Manual_Gaussian(_CVAE_Dist):
    """
    Roll-your-own implementation of a factorised Gaussian distribution---once
    again, this should be identical to the PyTorch wrapper above.
    """

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        mean, std = distutils.broadcast_all(mean, std)
        _CVAE_Dist.__init__(self, batch_shape=mean.size(), device=mean.device,
                            dtype=mean.dtype)
        self.mean = mean.to(self.device, self.dtype)
        self.std = std.to(self.device, self.dtype)

    def log_prob(self, value: Tensor) -> Tensor:
        var = self.std ** 2
        return -t.sum(0.5 * (
                        t.log(2. * math.pi * var) +
                        (value - self.mean)**2
                        / var)
                     , 1)

    def sample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        with t.no_grad():
            return t.normal(self.mean.expand(shape), self.std.expand(shape))
