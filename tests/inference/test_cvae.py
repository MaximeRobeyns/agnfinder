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

import agnfinder.inference as inf

from agnfinder.types import arch_t, Tensor
from agnfinder.inference import utils
from agnfinder.inference.base import CVAE, CVAEParams


# Image MNIST -----------------------------------------------------------------


class MNIST_img_params(CVAEParams):
    cond_dim = 10  # x; dimension of one-hot labels
    data_dim = 28*28  # y; size of MNIST image
    latent_dim = 2  # z

    prior = inf.StandardGaussianPrior
    prior_arch = None

    encoder = inf.FactorisedGaussianEncoder
    enc_arch = arch_t([data_dim + cond_dim, 256], [latent_dim, latent_dim],
                      nn.ReLU())

    decoder = inf.FactorisedGaussianDecoder
    dec_arch = arch_t([latent_dim + cond_dim, 256], [data_dim, data_dim],
                      nn.ReLU(), [nn.Sigmoid(), nn.ReLU()])


class MNIST_img_cvae(CVAE):

    def preprocess(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        if x.dim() > 2:
            x = x.view(-1, 28*28)
        switched_x = utils._onehot(y, 10).to(self.device, self.dtype)
        switched_y = x.to(self.device, self.dtype)
        return switched_x, switched_y


# Label MNIST -----------------------------------------------------------------


class MNIST_label_params(CVAEParams):
    cond_dim = 28*28  # x; dimension of MNIST image pixel data
    data_dim = 10  # y; size of one-hot encoded digit labels
    latent_dim = 2  # z

    prior = inf.StandardGaussianPrior
    prior_arch = None

    encoder = inf.FactorisedGaussianEncoder
    enc_arch = arch_t([data_dim + cond_dim, 256], [latent_dim, latent_dim],
                       nn.ReLU())

    decoder = inf.MultinomialDecoder
    dec_arch = arch_t([latent_dim + cond_dim, 256], [data_dim],
                       nn.ReLU(), [nn.Softmax(dim=1)])


class MNIST_label_cvae(CVAE):

    def preprocess(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        if x.dim() > 2:
            x = x.view(-1, 28*28)
        x = x.to(self.device, self.dtype)
        y = utils._onehot(y, 10).to(self.device, self.dtype)
        return x, y


# Testing ---------------------------------------------------------------------


def test_cvae_params():
    """Tests assertions ('type safety') during CVAE parameter definition"""

    with pytest.raises(ValueError):
        # input of prior net != cond_dim
        class P1(CVAEParams):
            cond_dim = 1
            data_dim = 1
            latent_dim = 1
            prior = inf.StandardGaussianPrior
            prior_arch = arch_t([2, 1], [1], nn.ReLU())
            encoder = inf.FactorisedGaussianEncoder
            enc_arch = arch_t([2, 1], [1], nn.ReLU())
            decoder = inf.FactorisedGaussianDecoder
            dec_arch = arch_t([2, 1], [1], nn.ReLU())
        _ = P1()

    with pytest.raises(ValueError):
        # input of enc net != data_dim + cond_dim
        class P2(CVAEParams):
            cond_dim = 1
            data_dim = 1
            latent_dim = 1
            prior = inf.StandardGaussianPrior
            prior_arch = None
            encoder = inf.FactorisedGaussianEncoder
            enc_arch = arch_t([3, 1], [1, 1], nn.ReLU())
            decoder = inf.FactorisedGaussianDecoder
            dec_arch = arch_t([2, 1], [1, 1], nn.ReLU())
        _ = P2()

    with pytest.raises(ValueError):
        # input of dec net != latent_dim + cond_dim
        class P3(CVAEParams):
            cond_dim = 1
            data_dim = 1
            latent_dim = 1
            prior = inf.StandardGaussianPrior
            prior_arch = None
            encoder = inf.FactorisedGaussianEncoder
            enc_arch = arch_t([2, 1], [1, 1], nn.ReLU())
            decoder = inf.FactorisedGaussianDecoder
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
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    ip = MNIST_img_params()
    _ = MNIST_img_cvae(ip, device=device, dtype=t.float64)

    lp = MNIST_label_params()
    _ = MNIST_label_cvae(lp, device=device, dtype=t.float64)

    assert True  # haven't fallen over yet, great success!!


@pytest.mark.slow
def test_cvae_MNIST_img():
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    ip = MNIST_img_params()
    cvae = MNIST_img_cvae(ip, device=device, dtype=t.float64)
    train_loader, test_loader = utils._load_mnist()

    initial_loss = cvae.test_generator(test_loader)
    print('done initial loss')
    # train for just 1 epoch to keep things speedy
    cvae.trainmodel(train_loader, epochs=1)
    print('done train model')
    final_loss = cvae.test_generator(test_loader)
    print('done final loss')

    # Assert that loss is lower after a spot of training...
    # This isn't *hugely* informative, but it's good enough for a basic test
    assert initial_loss > final_loss

    # Perhaps generate samples of images from [0..9], and save to image
    # for visual inspection. Also write date / time in title of plot

@pytest.mark.slow
def test_cvae_MNIST_label():
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    lp = MNIST_label_params()
    cvae = MNIST_label_cvae(lp, device=device, dtype=t.float64)
    train_loader, test_loader = utils._load_mnist()

    initial_loss = cvae.test_generator(test_loader)
    # train for just 1 epoch to keep things speedy(ish)
    cvae.trainmodel(train_loader, epochs=1)
    final_loss = cvae.test_generator(test_loader)

    # Assert that loss is lower after a spot of training...
    # This isn't *hugely* informative, but it's good enough for a basic test
    assert initial_loss > final_loss
