# AGNfinder: Detect AGN from photometry in XXL data.
#
# Copyright (C) 2021 Maxime Robeyns <maximerobeyns@gmail.com>
# Copyright (C) 2019-20 Mike Walmsley <walmsleymk1@gmail.com>
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
"""Estimate parameter PDFs using MCMC methods from into prospector.
"""

import logging
import datetime
import torch as t
import pandas as pd
import numpy as np

from typing import Callable
from torch.utils.data import DataLoader
from prospect.fitting import lnprobfn
from prospect.fitting import fit_model
from multiprocessing import Pool

import agnfinder.config as cfg

from agnfinder.types import Tensor, tensor_like
from agnfinder.prospector import Prospector
from agnfinder.simulation.utils import normalise_theta
from agnfinder.types import Filters, prun_params_t, \
                            column_order, EMCEE, Dynesty
from agnfinder.prospector.load_photometry import load_galaxy
from agnfinder.inference.inference import Model, InferenceParams
from agnfinder.inference.mcmc_util import MCMCParams


# Main MCMC Class --------------------------------------------------------------
# We extend the `Model` class, which is more suited to machine learning
# models (i.e. the presence of the `trainmodel` abstract method), in order to
# keep things consistent and use the same interfaces.

class MCMC(Model):

    def __init__(self, mp: MCMCParams, overwrite_results: bool = False,
                 logging_callbacks: list[Callable] = []):

        self.device = mp.device
        self.dtype = mp.dtype
        self.filters = mp.filters
        self.emulate_ssp = mp.emulate_ssp
        self.catalogue_loc = mp.catalogue_loc
        self.inference_procedure = mp.inference_procedure

        self.overwrite_results = overwrite_results
        self.logging_callbacks = logging_callbacks

    name: str = 'MCMC'

    def __repr__(self) -> str:
        r = f'{self.inference_procedure} {self.name} '
        if self.inference_procedure == Dynesty:
            # TODO list out relevant Dynesty parameters
            r += f' with ...'
        elif self.inference_procedure == EMCEE:
            # TODO list out relevant emcee parameters
            r += f' with ...'
        return r

    def fpath(self) -> str:
        """Returns a file path to save the model to, based on its parameters."""
        raise AttributeError("MCMC doesn't support saving models.")

    def trainmodel(self, train_loader: DataLoader, ip: InferenceParams,
                   *args, **kwargs) -> None:
        raise AttributeError("MCMC doesn't support `trainmodel`")

    def sample(self, x: tensor_like, n_samples: int = 1000, *args, **kwargs) -> Tensor:

        # Since we need to retain the column names, we require either a structured
        # numpy array, or better still, the original pd.Series
        assert isinstance(x, pd.Series)
        galaxy = x

        print('galaxy at this point is:')
        print(galaxy)

        p = Prospector(self.filters, self.emulate_ssp, galaxy)

        start_time = datetime.datetime.now()
        logging.info(f'Begining MCMC ({self.inference_procedure}) sample at {start_time}')

        if self.inference_procedure == Dynesty:
            # TODO verify that these are indeed denormalised.
            denormed_samples = self._dynesty_galaxy(p, n_samples)
        else:
            assert self.inference_procedure == EMCEE
            denormed_samples = self._mcmc_galaxy(p, n_samples)
        assert denormed_samples is not None

        duration = datetime.datetime.now() - start_time
        logging.info(f'{self.inference_procedure} sampling took {duration.seconds} seconds')

        # normalise the samples
        zind = column_order.index('redshift')
        lims_without_z = column_order[:zind] + column_order[zind+1:]
        norm_samples = normalise_theta(
                t.Tensor(denormed_samples), cfg.FreeParams(lims_without_z))

        return norm_samples

    def _mcmc_galaxy(self, p: Prospector, n_samples: int) -> np.ndarray:
        params = cfg.EMCEEParams()
        logging.info((
            f'Running EMCEE with walkers: {params.nwalkers}, iterations: '
            f'{params.niter}, burn-in: {params.nburn}'))

        run_params: prun_params_t = p.run_params | {
            'optimize': True,
            'emcee': True,
            'dynesty': False,
            'nwalkers': params.nwalkers,
            # 'niter': params.niter,
            'niter': n_samples,
            'nburn': params.nburn,
        }

        output = fit_model(p.obs, p.model, p.sps, lnprobfn=lnprobfn, **run_params)

        return output['sampling'][0].samples

    def _dynesty_galaxy(self, p: Prospector, n_samples: int) -> np.ndarray:
        params = cfg.DynestyParams()
        logging.info(f'Running Dynesty (nested) sampling')

        run_params: prun_params_t = p.run_params | {
            'dynesty': True,
            'optimize': False,
            'emcee': False,
        }

        dp = cfg.DynestyParams()
        # TODO figure out which of these parameters controls the number of samples.
        dynesty_params: prun_params_t = {
            'nested_method': dp.method,
            'nlive_init': dp.nlive_init,
            'nlive_batch': dp.nlive_batch,
            'nested_dlogz_init': dp.dlogz_init,
            'nested_posterior_thresh': dp.posterior_thresh,
            'nested_maxcall': dp.maxcall
        }
        run_params |= dynesty_params

        output = fit_model(p.obs, p.model, p.sps, lnprobfn=lnprobfn, **run_params)
        return output['sampling'][0].samples

    def _save_samples(self, samples: np.ndarray):
        raise NotImplementedError("TODO: implement saving MCMC samples to disk.")

if __name__ == '__main__':
    cfg.configure_logging()

    ip = cfg.InferenceParams()
    mcmcp = cfg.MCMCParams()

    mcmc = MCMC(mcmcp)

    logging.info('Successfully initialised mcmc class')

    # optionally provide an index for a specific galaxy.
    # index = 0
    index = None

    # catalogue: pd.DataFrame = load_catalogue(ip.catalogue_loc, Filters.DES)
    galaxy, idx = load_galaxy(ip.catalogue_loc, Filters.DES, index)
    mcmc.sample(galaxy, 10000)

    logging.info('MCMC sampling')
