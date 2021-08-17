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

    def distribution(self, y: Tensor, x: Tensor) -> Distribution:
        """
        Takes the physical parameters y and photometric observations x, and
        returns a distribution over latents.
        """
        params = self.forward(t.cat((y, x), -1))
        assert self.out_len == 2
        assert len(params) == self.out_len
        d = dist.MultivariateNormal(params[0], t.diag(t.pow(params[1], 2.)))

        assert d.has_rsample
        return d

class Prior(base.PriorNet):

    def distribution(self, x: Tensor) -> Distribution:
        """
        Accepts conditioning information x (photometry), and returns a
        distribution over latent vectors.
        """
        params = self.forward(x)
        assert self.out_len == 2
        assert len(params) == self.out_len
        d = dist.MultivariateNormal(params[0], t.diag(t.pow(params[1], 2.)))
        return d

class Generator(base.GeneratorNet):

    def distribution(self, z: Tensor, x: Tensor) -> Distribution:
        """
        Accepts a latent vector z as well as conditioning information x, and
        returns a distribution over outputs y.
        """
        params = self.forward(t.cat((z, x), -1))
        assert len(params) == self.out_len
        assert self.out_len == 2
        d = dist.MultivariateNormal(params[0], t.diag(t.pow(params[1], 2.)))
        return d


# if required, create a base.CVAE class
class CVAE(object):
    """Conditional VAE class.

    TODO: implement methods such as encode, decode, sapmle, generate, forward
    and loss_function.

    Training procedure:
    for e in epochs:
        iterate through training pairs (y, x)
        where y = galaxy params (9 dims), and x = photometric obs (8 dims)
        1. use recognition net to output params of q distribution, so we can
           use reparametrised sampling to get a z sample. Keep track of the
           epsilon used, or return log q_{phi}(z' | y, x) then and there along
           with the z sample.
        2. use prior network to get params of prior distribution p_{theta}(z | x)
        3. use decoder / generator network to map sampled z and obs x to params
           over y

        Calculate ELBO objective to be maximised by evaluating likelihoods of
        the various distributions.
        - You only need us use the special reparametrised sampling density
          equation for the recognition distribution q_{phi}(.); for the others
          you can just do log_prob.
    """

    def __init__(self, recognition_arch: arch_t, prior_arch: arch_t,
                 generator_arch: arch_t) -> None:

        self.recognition_net = Recognition(recognition_arch)
        self.prior_net = Prior(prior_arch)
        self.generator_net = Generator(generator_arch)

        self.phi_opt = t.optim.Adam(self.recognition_net.parameters())
        # TODO we perhaps need to have another one here
        self.theta_opt = t.optim.Adam(self.prior_net.parameters())


    def train(self, train_loader: DataLoader, epochs: int = 2):
        for _ in range(epochs):
            pass

    def loss_function(self):
        # TODO implement this
        # reconstruction loss + KL divergence
        pass


if __name__ == '__main__':

    cfg.configure_logging()

    ip = cfg.InferenceParams()  # inference procedure parameters
    cp = cfg.CVAEParams()  # CVAE model hyperparameters

    # initialise the model
    cvae = CVAE(cp.recognition_arch, cp.prior_arch, cp.generator_arch)

    # load the generated (theta, photometry) dataset
    train_loader, test_loader = load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=ip.batch_size)

    # train the CVAE
    cvae.train(train_loader, ip.epochs)

    # TODO evaluate and optionally output plots and figures



