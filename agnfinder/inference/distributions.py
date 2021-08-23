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
    """Reparametrised univariate Gaussian distribution

    Here we simply make use of PyTorch's existing Normal distribution, which
    implements the abstract methods required by _CVAE_RDist: log_prob and
    rsample.
    """

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        super(_CVAE_Dist, self).__init__()
        mean = mean.to(self.device, self.dtype)
        std = std.to(self.device, self.dtype)
        super(dist.Normal, self).__init__(mean, std)
        assert self.has_rsample


# Simple distributions ========================================================


class Gaussian(_CVAE_Dist, dist.Normal):
    """Univariate Gaussian distribution"""

    def __init__(self, mean: Tensor, log_std: Tensor) -> None:
        super(_CVAE_Dist, self).__init__()
        super(dist.Normal, self).__init__(mean, t.exp(log_std))
