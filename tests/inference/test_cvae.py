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

import math
import pytest
import torch as t
import torch.nn as nn

from typing import Optional, Union
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.profiler import profile, record_function, ProfilerActivity

import agnfinder.inference.distributions as dist

from agnfinder.types import arch_t, DistParams, Tensor
from agnfinder.inference.base import CVAE, CVAEParams, \
                                     CVAEDec, CVAEEnc, CVAEPrior, \
                                     _CVAE_Dist, _CVAE_RDist


# testing utilities -----------------------------------------------------------


def load_mnist(batch_size: int = 64, dtype: t.dtype = t.float64,
               device: t.device = t.device('cpu')
               ) -> tuple[DataLoader, DataLoader]:
    """(Down)load MNIST dataset in ./data/testdata, and return training and
    test DataLoaders using specified batch_size.
    """

    cuda_kwargs = {'num_workers': 1}  # , 'pin_memory': True}
    train_kwargs = {'batch_size': batch_size, 'shuffle': True} | cuda_kwargs
    test_kwargs = {'batch_size': batch_size, 'shuffle': False} | cuda_kwargs

    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.to(device, dtype)
    ])

    train_set = datasets.MNIST('./data/testdata', train=True, download=True,
                               transform=transform)
    test_set = datasets.MNIST('./data/testdata', train=False, download=True,
                              transform=transform)

    train_loader = DataLoader(train_set, **train_kwargs)
    test_loader = DataLoader(test_set, **test_kwargs)

    return train_loader, test_loader


def onehot(idx: Tensor, n: int) -> Tensor:
    """Turns an index into a one-hot encoded vector, of length n"""
    assert t.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = t.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot


# Testing MNIST CVAE definition -----------------------------------------------
# x = 10 dimensional one-hot encoded image labels ('conditioning info')
# y = 28*28 = 784 dimensional pixel data ('data')
# z = 2 dimensional latent vector


class StandardGaussianPrior(CVAEPrior):
    def get_dist(self, dist_params=None) -> _CVAE_Dist:
        return dist.Gaussian(t.zeros(1), t.ones(1))

class GaussianEncoder(CVAEEnc):
    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_RDist:
        assert isinstance(dist_params, list)
        return dist.R_Gaussian(dist_params[0], t.exp(dist_params[1]))


class GaussianDecoder(CVAEDec):
    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_Dist:
        assert isinstance(dist_params, list)
        return dist.Gaussian(dist_params[0], t.exp(dist_params[1]))


class MNIST_img_params(CVAEParams):
    cond_dim = 10  # x; dimension of one-hot labels
    data_dim = 28*28  # y; size of MNIST image
    latent_dim = 2  # z

    # Gaussian prior network p_{theta}(z | x)
    prior = StandardGaussianPrior
    prior_arch = None

    encoder = GaussianEncoder
    enc_arch = arch_t([data_dim + cond_dim, 256], [latent_dim, latent_dim],
                      nn.ReLU(), batch_norm=False)

    decoder = GaussianDecoder
    dec_arch = arch_t([latent_dim + cond_dim, 256], [data_dim, data_dim],
                      nn.ReLU(), [nn.Sigmoid(), None], batch_norm=False)


class MNIST_img_cvae(CVAE):

    def preprocess(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        # for the inputs, x is pixel data, and y are integer MNIST labels.
        #
        # But here we want x to be one-hot encoded labels, and y to be pixel
        # data. So we apply this pre-processing, and switch the order when we
        # return.

        if x.dim() > 2:
            x = x.view(-1, 28*28)

        switched_x = onehot(y, 10).to(self.device, self.dtype)
        switched_y = x.to(self.device, self.dtype)
        return switched_x, switched_y


# Testing ---------------------------------------------------------------------


def test_cvae_params():
    """Tests assertions ('type safety') during CVAE parameter definition"""

    with pytest.raises(ValueError):
        # input of prior net != cond_dim
        class P1(CVAEParams):
            cond_dim = 1
            data_dim = 1
            latent_dim = 1
            prior = StandardGaussianPrior
            prior_arch = arch_t([2, 1], [1], nn.ReLU())
            encoder = GaussianEncoder
            enc_arch = arch_t([2, 1], [1], nn.ReLU())
            decoder = GaussianDecoder
            dec_arch = arch_t([2, 1], [1], nn.ReLU())
        _ = P1()

    with pytest.raises(ValueError):
        # input of enc net != data_dim + cond_dim
        class P2(CVAEParams):
            cond_dim = 1
            data_dim = 1
            latent_dim = 1
            prior = StandardGaussianPrior
            prior_arch = None
            encoder = GaussianEncoder
            enc_arch = arch_t([3, 1], [1, 1], nn.ReLU())
            decoder = GaussianDecoder
            dec_arch = arch_t([2, 1], [1, 1], nn.ReLU())
        _ = P2()

    with pytest.raises(ValueError):
        # input of dec net != latent_dim + cond_dim
        class P3(CVAEParams):
            cond_dim = 1
            data_dim = 1
            latent_dim = 1
            prior = StandardGaussianPrior
            prior_arch = None
            encoder = GaussianEncoder
            enc_arch = arch_t([2, 1], [1, 1], nn.ReLU())
            decoder = GaussianDecoder
            dec_arch = arch_t([3, 1], [1, 1], nn.ReLU())
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

    p = MNIST_img_params()

    _ = MNIST_img_cvae(p, device=t.device('cpu'), dtype=t.float64)

    assert True  # haven't fallen over yet, great success!!


@pytest.mark.slow
def test_cvae_MNIST():
    p = MNIST_img_params()
    cvae = MNIST_img_cvae(p, device=t.device('cpu'), dtype=t.float64)
    train_loader, _ = load_mnist()

    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True, use_cuda=True) as prof:
    cvae.trainmodel(train_loader, epochs=1)

    # generate samples of images from [0..9], and save to image for visual inspection.
    # also write date / time in title of plot

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
