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
from typing import Optional

import agnfinder.inference.base as base

from agnfinder import config as cfg
from agnfinder.types import Tensor, Distribution, DistParam, CVAEParams
from agnfinder.inference.utils import load_simulated_data


class CVAE(base.CVAE):
    """Basic initial CVAE implementation using isotropic Gaussians for all
    distributions"""

    def recognition_params(self, y: Tensor, x: Tensor) -> DistParam:
        in_vec = t.cat((y, x), -1)
        params = self.recognition_net(in_vec)
        assert isinstance(params, list)
        # second head outputs log covariance, so we exponentiate before returning
        params[1] = t.exp(params[1])
        return params

    def prior(self, x: Tensor) -> Distribution:
        params = self.prior_net(x)
        assert len(params) == self.prior_net.out_len
        assert self.prior_net.out_len == 2
        [batch, latent_dim] = params[1].shape
        I = t.eye(latent_dim).expand(batch, -1, -1)
        cov = I * t.exp(params[1]).unsqueeze(-1)

        d = dist.MultivariateNormal(params[0], cov)
        return d

    def generator(self, z: Tensor, x: Tensor) -> Distribution:
        params = self.generator_net(t.cat((z, x), -1))
        assert len(params) == self.generator_net.out_len
        assert self.generator_net.out_len == 2
        [batch, latent_dim] = params[1].shape
        I = t.eye(latent_dim).expand(batch, -1, -1)
        cov = I * t.exp(params[1]).unsqueeze(-1)
        d = dist.MultivariateNormal(params[0], cov)
        return d

    def rsample(self, y: Tensor, x: Tensor) -> tuple[Tensor, DistParam]:
        [mu, cov] = self.recognition_params(y, x)
        eps = self.EKS.sample(cov.shape[0])
        z = mu + cov * eps
        return z, [eps, mu, cov]

    def kl_div(self, z: Tensor, x: Tensor, rparams: DistParam) -> Tensor:
        [eps, cov, _] = rparams
        logqz = self.EKS.log_prob(eps) - t.log(cov).sum(1)

        prior_dist = self.prior(x)
        logpz = prior_dist.log_prob(z)

        return logpz + logqz

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
    logging.info('Trained CVAE')

    # TODO evaluate and optionally output plots and figures
