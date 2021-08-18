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
"""Tests for the conditional variational autoencoder.

To verify the basic functionality of the implementation, we train and run the
CVAE methods on MNIST.

Need to test
- initialisation (model initialises correctly; no errors)
- training (loss decreases and eventually converges)
- inference (handwritten digit recognition)
- generation (conditional generation of digit given label)
"""

import pytest
import torch as t
import torch.nn as nn

from agnfinder.types import arch_t, CVAEParams
from agnfinder.inference.inference import CVAE


class CVAEParams_testing(CVAEParams):
    cond_dim: int = 10  # x; dimension of one-hot labels
    data_dim: int = 28*28  # y; size of MNIST image
    latent_dim: int = 2  # z

    # Gaussian recognition model q_{phi}(z | y, x)
    recognition_arch: arch_t = arch_t(
        layer_sizes=[data_dim + cond_dim, 256],
        activations=nn.ReLU(),
        head_sizes=[latent_dim, latent_dim],
        head_activations=None,
        batch_norm=True)

    # (conditional) Gaussian prior network p_{theta}(z | x)
    prior_arch: arch_t = arch_t(
        layer_sizes=[cond_dim, 256],
        activations=nn.ReLU(),
        head_sizes=[latent_dim, latent_dim],
        head_activations=None,
        batch_norm=True)

    # generator network arch: p_{theta}(y | z, x)
    # Assume Gaussian parameters
    generator_arch: arch_t = arch_t(
        layer_sizes=[latent_dim + cond_dim, 256],
        activations=nn.ReLU(),
        head_sizes=[data_dim, data_dim],
        head_activations=None,
        batch_norm=True)


def test_cvae_params():
    """Tests assertions ('type safety') during CVAE parameter definition"""

    with pytest.raises(ValueError):
        # input of rec net != data_dim + cond_dim
        class P1(CVAEParams):
            cond_dim = 1
            data_dim = 1
            latent_dim = 1
            recognition_arch = arch_t([3, 1], [1], nn.ReLU())
            prior_arch = arch_t([1, 1], [1], nn.ReLU())
            generator_arch = arch_t([2, 1], [1], nn.ReLU())
        _ = P1()

    with pytest.raises(ValueError):
        # input of prior net != cond_dim
        class P2(CVAEParams):
            cond_dim = 1
            data_dim = 1
            latent_dim = 1
            recognition_arch = arch_t([2, 1], [1], nn.ReLU())
            prior_arch = arch_t([2, 1], [1], nn.ReLU())
            generator_arch = arch_t([2, 1], [1], nn.ReLU())
        _ = P2()

    with pytest.raises(ValueError):
        # input of gen net != latent_dim + cond_dim
        class P3(CVAEParams):
            cond_dim = 1
            data_dim = 1
            latent_dim = 1
            recognition_arch = arch_t([2, 1], [1], nn.ReLU())
            prior_arch = arch_t([1, 1], [1], nn.ReLU())
            generator_arch = arch_t([3, 1], [1], nn.ReLU())
        _ = P3()

    with pytest.raises(NotImplementedError):
        # missing abstract property
        class P4(CVAEParams):
            cond_dim = 1
            data_dim = 1
            latent_dim = 1
            recognition_arch = arch_t([2, 1], [1], nn.ReLU())
            prior_arch = arch_t([1, 1], [1], nn.ReLU())
        _ = P4()


def test_cvae_initialisation():
    """Tests that we can initialise a CVAE"""

    p = CVAEParams_testing()

    _ = CVAE(p, device=t.device('cpu'), dtype=t.float64)

    assert True  # haven't fallen over yet, great success!!
