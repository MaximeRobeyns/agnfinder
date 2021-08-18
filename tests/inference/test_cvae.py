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
import torch.distributions as dist

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from agnfinder.types import arch_t, CVAEParams, DistParam, Distribution, Tensor
from agnfinder.inference.base import CVAE


# testing utilities -----------------------------------------------------------


def load_mnist(batch_size: int = 64, dtype: t.dtype = t.float64,
        device: t.device = t.device('cpu')) -> tuple[DataLoader, DataLoader]:
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

class MNIST_img_params(CVAEParams):
    cond_dim: int = 10  # x; dimension of one-hot labels
    data_dim: int = 28*28  # y; size of MNIST image
    latent_dim: int = 2  # z

    # Gaussian recognition model q_{phi}(z | y, x)
    recognition_arch: arch_t = arch_t(
            [data_dim + cond_dim, 256], [latent_dim, latent_dim], nn.ReLU(), batch_norm=False)

    # (conditional) Gaussian prior network p_{theta}(z | x)
    prior_arch: arch_t = arch_t(
        [cond_dim, 256], [latent_dim, latent_dim], nn.ReLU(), batch_norm=False)

    # generator network arch: p_{theta}(y | z, x)
    generator_arch: arch_t = arch_t(
        [latent_dim + cond_dim, 256], [data_dim, data_dim], nn.ReLU(),
        [nn.Sigmoid(), None], batch_norm=False)


class MNIST_img_cvae(CVAE):
    """
    We may either treat x (the conditioning information) as the MNIST labels,
    and y as the images themselves, or vice versa.

    Here we take the former view (where x is a label, and y is the pixel data).
    At test time, we can therefore sample some z from a standard Gaussian,
    provide a desired image label, x, and generate plausible looking 28x28
    pixel images with a Gaussian likelihood.
    """

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

    def recognition_params(self, y: Tensor, x: Tensor) -> DistParam:
        params = self.recognition_net(t.cat((x, y), -1))
        params[1] = t.exp(params[1])
        return params

    def prior(self, x: Tensor) -> Distribution:
        """Isotropic Gaussian prior"""
        params = self.prior_net(x)
        [batch, latent_dim] = params[1].shape
        I = t.eye(latent_dim, device=self.device, dtype=self.dtype)
        cov = I.expand(batch, -1, -1) * t.exp(params[1]).unsqueeze(-1)
        return dist.MultivariateNormal(params[0], cov)

    def generator(self, z: Tensor, x: Tensor) -> Distribution:
        params = self.generator_net(t.cat((z, x), -1))
        [batch, output_dim] = params[1].shape
        I = t.eye(output_dim, device=self.device, dtype=self.dtype)
        cov = I.expand(batch, -1, -1) * t.exp(params[1]).unsqueeze(-1)
        return dist.MultivariateNormal(params[0], cov)

    def rsample(self, y: Tensor, x: Tensor) -> tuple[Tensor, DistParam]:
        [mu, cov] = self.recognition_params(y, x)
        eps = self.EKS.sample((cov.shape[0],))
        z = mu + cov * eps
        return z, [eps, mu, cov]

    def kl_div(self, z: Tensor, x: Tensor, rparams: DistParam) -> Tensor:
        # x is 10 dimensional
        [eps, _, cov] = rparams
        logqz = self.EKS.log_prob(eps) - t.log(cov).sum(1)

        prior_dist = self.prior(x)
        logpz = prior_dist.log_prob(z)

        return logpz + logqz


# Testing ---------------------------------------------------------------------


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

    p = MNIST_img_params()

    _ = MNIST_img_cvae(p, device=t.device('cpu'), dtype=t.float64)

    assert True  # haven't fallen over yet, great success!!


@pytest.mark.slow
def test_cvae_MNIST():
    p = MNIST_img_params()
    cvae = MNIST_img_cvae(p, device=t.device('cuda'), dtype=t.float64)
    train_loader, _ = load_mnist()
    cvae.trainmodel(train_loader, epochs=1)
    assert False
