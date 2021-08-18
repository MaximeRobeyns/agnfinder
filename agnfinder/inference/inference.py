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
from agnfinder.types import Tensor, Distribution, DistParam, CVAEParams
from agnfinder.inference.utils import load_simulated_data
from agnfinder.inference.base import EKS


class Recognition(base.RecognitionNet):
    """Gaussian q_{phi}(z | y, x) parameters"""

    def dist_params(self, y: Tensor, x: Tensor) -> DistParam:
        """
        Takes the physical parameters y and photometric observations x, and
        returns parameters for MVN over latents.
        """
        in_vec = t.cat((y, x), -1)
        params = self.forward(in_vec)
        assert isinstance(params, list)
        return params


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

    def __init__(self, cp: CVAEParams,
                 device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        """Initialise a conditional VAE.

        Args:
            cp: CVAE parameters (from config.py)
            recognition_arch: architecture for q_{phi}(z | y, x)
            prior_arch: architecture for p_{theta_z}(z | x)
            generator_arch: architecture for p_{theta_y}(y | z, x)
            device: device on which to store model
            dtype: data type to use in tensors
        """

        self.device = device
        self.dtype = dtype

        self.recognition_net = Recognition(cp.recognition_arch, device, dtype)
        self.prior_net = Prior(cp.prior_arch, device, dtype)
        self.generator_net = Generator(cp.generator_arch, device, dtype)

        # standard Gaussian for generating epsilon samples
        self.EKS = EKS(cp.latent_dim)

        # optimisers for the various network parameters
        self.phi_opt = t.optim.Adam(self.recognition_net.parameters())
        self.theta_z_opt = t.optim.Adam(self.prior_net.parameters())
        self.theta_y_opt = t.optim.Adam(self.generator_net.parameters())
        # list for convenience (mapping)
        self.opts = [self.phi_opt, self.theta_z_opt, self.theta_y_opt]

    def train(self, train_loader: DataLoader, epochs: int = 2,
              log_every: int = 100):
        """Training procedure for CVAE

        1. Encode an (y, x) pair using q (recognition network), to obtain
           mean and covariance of an isotropic Gaussian distribution;
           (mu, L) parametrises q_{phi}(z | y, x).
        2. Sample epsilon from p(epsilon) (EKS; standard Gaussian).
        3. Compute z = mu + L * epsilon.
        4. Compute density of z under q_{phi}(z | y, x); using the
           change-of-variables formula:

            log q_{phi}(z | y, x) = log p(epsilon) - log d_{phi}(y, x, epsilon)

           where log d_{phi}(y, x, epsilon) is the log determinant of the
           Jacobian of the transformation.
        5. Compute log prob of (reparametrised) z sample (from step 3) under
           p_{theta_z}(z | x); by parametrising a MVN by prior_net:
           (mu, std) = f_{theta_z}(x), and calling log_prob(z).
        6. Compute (mu, std) = f_{theta_y}(z, x) to parametrise the Gaussian
           p_{theta}(y | z, x). Evaluate log_prob(y).
        7. Find ELBO (approx marginal likelihood) by summing

           (step 6) + (step 5) - (step 4)

        8. Convert approx marginal likelihood to loss with -ELBO, and use
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

                # 1.
                [mu, cov] = self.recognition_net.dist_params(y, x)

                # 2.
                eps = self.EKS.generate(t.Size((cov.shape[0],)))

                # 3.
                z = mu + cov * eps

                # 4.
                logqz = self.EKS.log_prob(eps) - cov.sum(1)

                # 5.
                prior_dist = self.prior_net.distribution(x)
                logpz = prior_dist.log_prob(z)

                # 6.
                gen_dist = self.generator_net.distribution(z, x)
                logpy = gen_dist.log_prob(y)

                #7.
                LL = logpy
                KL = logpz + logqz
                ELBO = LL - KL

                #8.
                loss = (-ELBO).mean(0)

                map(lambda opt: opt.zero_grad(), self.opts)
                loss.backward()
                map(lambda opt: opt.step(), self.opts)

                if i % log_every == 0 or i == len(train_loader)-1:
                    print("Epoch: {:02d}/{:02d}, Batch: {:03d}/{:d}, Loss {:9.4f}".format(
                        e, epochs, i, len(train_loader)-1, loss.item()))


if __name__ == '__main__':

    cfg.configure_logging()

    ip = cfg.InferenceParams()  # inference procedure parameters
    cp = cfg.CVAEParams()  # CVAE model hyperparameters

    # initialise the model
    cvae = CVAE(cp, device=ip.device, dtype=ip.dtype)
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



