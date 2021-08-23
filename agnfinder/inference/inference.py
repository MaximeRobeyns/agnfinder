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
import agnfinder.inference.distributions as dist

from typing import Union, Optional

from agnfinder import config as cfg
from agnfinder.types import Tensor, DistParams
from agnfinder.inference.utils import load_simulated_data
from agnfinder.inference.base import CVAEPrior, CVAEEnc, CVAEDec, \
                                     _CVAE_Dist, _CVAE_RDist


class StandardGaussianPrior(CVAEPrior):
    def get_dist(self, dist_params = None) -> _CVAE_Dist:
        return dist.Gaussian(t.zeros(1), t.ones(1))


class GaussianEncoder(CVAEEnc):
    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_RDist:
        assert isinstance(dist_params, list)
        return dist.R_Gaussian(dist_params[0], t.exp(dist_params[1]))


class GaussianDecoder(CVAEDec):
    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_Dist:
        assert isinstance(dist_params, list)
        return dist.Gaussian(dist_params[0], t.exp(dist_params[1]))


if __name__ == '__main__':

    cfg.configure_logging()

    ip = cfg.InferenceParams()  # inference procedure parameters
    cp = cfg.CVAEParams()  # CVAE model hyperparameters

    # initialise the model
    cvae = cp.model(cp, device=ip.device, dtype=ip.dtype)
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
    cvae.trainmodel(train_loader, ip.epochs)
    logging.info('Trained CVAE')

    # TODO evaluate and optionally output plots and figures
