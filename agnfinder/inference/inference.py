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

from typing import Union, Optional
from torchvision import transforms

import agnfinder.inference.distributions as dist
from agnfinder import config as cfg
from agnfinder.types import Tensor, DistParams
from agnfinder.inference.utils import load_simulated_data
from agnfinder.inference.base import CVAE, CVAEPrior, CVAEEnc, CVAEDec, \
                                     _CVAE_Dist, _CVAE_RDist


# Priors ----------------------------------------------------------------------

class StandardGaussianPrior(CVAEPrior):
    """
    A standard Gaussian distribution, whose dimension matches the length of the
    latent code z.
    """
    def get_dist(self, _: Optional[Union[Tensor, DistParams]]=None) -> _CVAE_Dist:
        mean = t.zeros(self.latent_dim, device=self.device, dtype=self.dtype)
        std = t.ones(self.latent_dim, device=self.device, dtype=self.dtype)
        return dist.Gaussian(mean, std)


# Encoders --------------------------------------------------------------------


class FactorisedGaussianEncoder(CVAEEnc):
    """
    A factorised Gaussian encoder; the distribution returned by this encoder
    implements reparametrised sampling and log_prob methods.

    The following is an example of a compatible architecture:

    >>> arch_t(layer_sizes=[data_dim + cond_dim, ...],
    ...        head_sizes=[latent_dim, latent_dim], # loc, scale
    ...        nn.SiLU())

    """
    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_RDist:
        assert isinstance(dist_params, list) and len(dist_params) == 2
        [mean, log_std] = dist_params
        std = t.exp(log_std)
        return dist.R_Gaussian(mean, std)


class GaussianEncoder(CVAEEnc):
    """A full-covariance Gaussian encoder

    Expects `dist_params` to be a list of three Tensors;
    1. `mean` giving the location (vector),
    2. `log_std` a vector which gives the elements on the main diagonal of the
        covariance matrix,
    3. `L` is a matrix which will be masked to a lower-triangular matrix before
        exp(log_std) is added; giving the final covariance matrix.

    Here is an example architecture:

    >>> arch_t(layer_sizes=[data_dim + cond_dim, ...],
    ...        head_sizes=[latent_dim, latent_dim, latent_dim*latent_dim],
    ...        nn.SiLU(), [None, nn.ReLU(), nn.ReLU()])

    """
    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_RDist:
        assert isinstance(dist_params, list) and len(dist_params) == 3
        [mean, log_std, L] = dist_params
        L = L.reshape((-1, log_std.size(-1), log_std.size(-1)))
        std_diag = t.exp(log_std).diag_embed()
        assert std_diag.shape == L.shape
        L = t.tril(L, -1) + std_diag
        return dist.R_MVN(mean, L)


# Decoders --------------------------------------------------------------------


class FactorisedGaussianDecoder(CVAEDec):
    """
    A factorised Gaussian decoder. This corresponds to using a Gaussian
    likelihood in ML training; or equivalently minimising an MSE loss between
    the target data and the reconstruction.

    Example compatible architecture:

    >>> arch = arch_t(layer_sizes=[latent_dim + cond_dim, ...], # input / hidden
    ...               head_sizes=[data_dim, data_dim], # loc and scale
    ...               activations=nn.SiLU(), # input / hidden layer activations
    ...               [None, Squareplus(1.2)]) # scale must be positive

    """
    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_Dist:
        assert isinstance(dist_params, list) and len(dist_params) == 2
        [mean, log_std] = dist_params
        std = t.exp(log_std)
        return dist.Gaussian(mean, std)


class MultinomialDecoder(CVAEDec):
    """
    A multinomial decoder. We shouldn't need this for galaxy data, but it's
    here for tests and notebooks using MNIST.

    Example architecture is:

    >>> arch = arch_t(layer_sizes=[latent_dim + cond_dim, ...],
    ...               head_sizes=[data_dim],  # p parameter for multinomial
    ...               activations=SiLU(),
    ...               head_activations=[nn.Softmax()]) # relative probs

    """
    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_Dist:
        assert isinstance(dist_params, t.Tensor)
        return dist.Multinomial(dist_params)


class LaplaceDecoder(CVAEDec):
    """
    A Laplace likelihood.

    Example compatible architecture (similar to factorised Gaussian models):

    >>> arch = arch_t(layer_sizes=[latent_dim + cond_dim, ...], # input / hidden
    ...               head_sizes=[data_dim, data_dim], # loc and scale
    ...               activations=nn.SiLU(), # input / hidden layer activations
    ...               [None, Squareplus(1.2)]) # scale must be positive

    """
    def get_dist(self, dist_params: Union[t.Tensor, DistParams]) -> _CVAE_Dist:
        assert isinstance(dist_params, list) and len(dist_params) == 2
        [loc, log_scale] = dist_params
        scale = t.exp(log_scale)
        return dist.Laplace(loc, scale)


if __name__ == '__main__':

    cfg.configure_logging()

    ip = cfg.InferenceParams()  # inference procedure parameters
    cp = cfg.CVAEParams()  # CVAE model hyperparameters

    # initialise the model
    cvae = ip.model(cp, device=ip.device, dtype=ip.dtype)
    logging.info('Initialised CVAE network')
    logging.info(f'Saving to: {cvae.fpath()}')

    # load the generated (theta, photometry) dataset
    train_loader, test_loader = load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=ip.batch_size,
        transforms=[
            transforms.ToTensor()
        ])
    logging.info('Created data loaders')

    # train the CVAE
    cvae.trainmodel(train_loader, ip.epochs)
    logging.info('Trained CVAE')

    t.save(cvae, cvae.fpath())
    logging.info(f'Saved model CVAE as: {cvae.fpath()}')

