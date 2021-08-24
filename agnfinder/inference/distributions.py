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

import torch as t
import torch.distributions as dist

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
        mean = mean.to(self.device, self.dtype)
        std = std.to(self.device, self.dtype)
        dist.Normal.__init__(self, mean, std)
        assert self.has_rsample

    def log_prob(self, value: Tensor) -> Tensor:
        return dist.Normal.log_prob(self, value).sum(-1)

    def rsample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        return dist.Normal.rsample(self, sample_shape)


# Simple distributions ========================================================


class Gaussian(_CVAE_Dist, dist.Normal):
    """Factorised Gaussian distribution"""

    def __init__(self, mean: Tensor, log_std: Tensor) -> None:
        _CVAE_Dist.__init__(self)
        dist.Normal.__init__(self, mean, t.exp(log_std))

    def log_prob(self, value: Tensor) -> Tensor:
        return dist.Normal.log_prob(self, value).sum(-1)

    def sample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        return dist.Normal.sample(self, sample_shape)
