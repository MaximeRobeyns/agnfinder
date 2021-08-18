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
"""Entrypoint for inference tasks."""

import logging
import torch as t
import torch.distributions as dist
from torch.utils.data import DataLoader

import agnfinder.inference.base as base

from agnfinder import config as cfg
from agnfinder.types import arch_t, Tensor, Distribution
from agnfinder.inference.utils import load_simulated_data


class Recognition(base.RecognitionNet):
    """Gaussian q_{phi}(z | y, x)"""

    def distribution(self, y: Tensor, x: Tensor) -> Distribution:
        """
        Takes the physical parameters y and photometric observations x, and
        returns a distribution over latents.
        """
        in_vec = t.cat((y, x), -1)
        params = self.forward(in_vec)

        # TODO move this to constructor
        assert self.out_len == 2
        assert len(params) == self.out_len
        assert params[0].shape == params[1].shape
        logging.debug(f'{self}')  # print NN architecture

        # batch of covariance matrices will be [batch_size, latent_dim, latent_dim]
        [batch, latent_dim] = params[1].shape
        I = t.eye(latent_dim).expand(batch, -1, -1)
        cov = I * t.pow(params[1], 2.).unsqueeze(-1)

        d = dist.MultivariateNormal(params[0], cov)

        assert d.has_rsample
        return d


class Prior(base.PriorNet):
    """Gaussian p_{theta}(z | x)"""

    def distribution(self, x: Tensor) -> Distribution:
        """
        Accepts conditioning information x (photometry), and returns a
        distribution over latent vectors.
        """
        params = self.forward(x)
        assert self.out_len == 2
        assert len(params) == self.out_len
        [batch, latent_dim] = params[1].shape
        I = t.eye(latent_dim).expand(batch, -1, -1)
        cov = I * t.pow(params[1], 2.).unsqueeze(-1)

        d = dist.MultivariateNormal(params[0], cov)
        return d


class Generator(base.GeneratorNet):
    """Gaussian p_{theta}(y | z, x)"""

    def distribution(self, z: Tensor, x: Tensor) -> Distribution:
        """
        Accepts a latent vector z as well as conditioning information x, and
        returns a distribution over outputs y.
        """
        params = self.forward(t.cat((z, x), -1))
        assert len(params) == self.out_len
        assert self.out_len == 2
        [batch, latent_dim] = params[1].shape
        I = t.eye(latent_dim).expand(batch, -1, -1)
        cov = I * t.pow(params[1], 2.).unsqueeze(-1)

        d = dist.MultivariateNormal(params[0], cov)
        return d


# if required, create a base.CVAE class
class CVAE(object):
    """Conditional VAE class.

    TODO: implement methods such as encode, decode, sapmle, generate, forward
    and loss_function.
    """

    def __init__(self, recognition_arch: arch_t, prior_arch: arch_t,
                 generator_arch: arch_t, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        """Initialise a conditional VAE.

        Args:
            recognition_arch: architecture for q_{phi}(z | y, x)
            prior_arch: architecture for p_{theta_z}(z | x)
            generator_arch: architecture for p_{theta_y}(y | z, x)
            device: device on which to store model
            dtype: data type to use in tensors
        """

        self.device = device
        self.dtype = dtype

        # TODO perform a set of assertions here to ensure that architectures
        # are compatible (e.g. checking dimensions of inputs / outputs)
        # TODO think about how these constraints could be enforced using types

        self.recognition_net = Recognition(recognition_arch, device, dtype)
        self.prior_net = Prior(prior_arch, device, dtype)
        self.generator_net = Generator(generator_arch, device, dtype)

        self.phi_opt = t.optim.Adam(self.recognition_net.parameters())
        self.theta_z_opt = t.optim.Adam(self.prior_net.parameters())
        self.theta_y_opt = t.optim.Adam(self.generator_net.parameters())

    # TODO abstract this as part of ABC?
    def train(self, train_loader: DataLoader, epochs: int = 2,
              log_every: int = 100):
        """Training procedure for CVAE

        1. Encode an (y, x) pair using q (recognition network), to obtain
           mean and covariance of an isotropic Gaussian distribution;
           (mu, L) parametrises p_{phi}(z | y, x).
        2. Sample epsilon from p(epsilon) (standard Gaussian).
        3. Compute z = mu + L * epsilon.
        4. Compute density of z under q_{phi}(z | y, x); using
           change-of-variables formula:

            log q_{phi}(z | y, x) = log p(epsilon) - log d_{phi}(y, x, epsilon)

           where log d_{phi}(y, x, epsilon) is the log determinant of the
           Jacobian of the reparametrisation.
        5. Compute log prob of (reparametrised) z sample (from step 3) under
           p_{theta_z}(z | x); by parametrising a MVN by prior_net:
           (mu, std) = f_{theta_z}(x), and calling log_prob(z).
        6. Compute (mu, std) = f_{theta_y}(z, x) to parametrise the Gaussian
           p_{theta}(y | z, x). Evaluate log_prob(y).
        7. Find ELBO (approx marginal likelihood) by summing

           (step 6) + (step 5) - (step 4)

        8. Convert approx marginal likelihood to a loss with -ELBO, and use
           computational graph to backprop (call .backward()). Update all
           network parameters by calling their respective optimisers' step()
           method.

        Args:
            train_loader: DataLoader containing (mini-batched) training set
            epochs: number of epochs for which to train
            log_every: frequency at which to log training information
        """
        for e in range(epochs):
            for i, (x, y) in enumerate(train_loader):

                assert x.dtype == t.float64
                assert y.dtype == t.float64

                qdist = self.recognition_net.distribution(y, x)
                z = qdist.rsample()
                pdist = self.prior_net.distribution(x)
                gendist = self.generator_net.distribution(z, x)

                self.phi_opt.zero_grad()
                self.theta_opt.zero_grad()

                # TODO fix this loss function.
                # implement something like: ELBO = self.loss_function()
                ELBO = gendist.log_prob(y) + pdist.log_prob(z) - qdist.log_prob(z)
                loss = -ELBO.mean(-1)
                loss.backward()

                self.phi_opt.step()
                self.theta_opt.step()

                if i % log_every == 0 or i == len(train_loader)-1:
                    print("Epoch: {:02d}/{:02d}, Batch: {:03d}/{:d}, Loss {:9.4f}".format(
                        e, epochs, i, len(train_loader)-1, loss.item()))


if __name__ == '__main__':

    cfg.configure_logging()

    ip = cfg.InferenceParams()  # inference procedure parameters
    cp = cfg.CVAEParams()  # CVAE model hyperparameters

    # initialise the model
    cvae = CVAE(cp.recognition_arch, cp.prior_arch, cp.generator_arch,
                device=ip.device, dtype=ip.dtype)
    logging.info('Initialised CVAE network')

    # load the generated (theta, photometry) dataset
    train_loader, test_loader = load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=ip.batch_size,
        transforms=[
            t.from_numpy,
            lambda x: x.to(dtype=ip.dtype, device=ip.device)
        ])
    logging.info('Created data loaders')

    # train the CVAE
    cvae.train(train_loader, ip.epochs)

    # TODO evaluate and optionally output plots and figures



