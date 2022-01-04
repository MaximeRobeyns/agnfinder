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
import ultranest

from abc import abstractmethod
from emcee import EnsembleSampler
from typing import Callable, Optional, Any, Type
from multiprocessing import Pool
from torch.utils.data import DataLoader
from prospect.fitting import lnprobfn
from prospect.fitting import fit_model
from prospect.sources import CSPSpecBasis
from prospect.models.sedmodel import SedModel

import agnfinder.config as cfg
import agnfinder.types as types

from agnfinder.types import MCMCMethod, Tensor, tensor_like, cpz_obs_dict_t, \
                            prun_params_t
from agnfinder.prospector import Prospector
from agnfinder.simulation.utils import denormalise_theta_np, normalise_theta
from agnfinder.prospector.load_photometry import sample_galaxies
from agnfinder.inference.inference import Model, InferenceParams
from agnfinder.inference.mcmc_util import MCMCParams, EMCEEParams, \
                                          DynestyParams, UltraNestParams


# Main MCMC Class --------------------------------------------------------------
# We extend the `Model` class, which is more suited to machine learning
# models (i.e. the presence of the `trainmodel` abstract method), in order to
# keep the interfaces consistent for the user.

class MCMC(Model):
    """A base MCMC model."""

    def __init__(self, mp: MCMCParams, overwrite_results: bool = False,
                 logging_callbacks: list[Callable] = []) -> None:

        self.device = mp.device
        self.dtype = mp.dtype
        self.filters = mp.filters
        self.emulate_ssp = mp.emulate_ssp
        self.catalogue_loc = mp.catalogue_loc

        # for compatability with parent model class:
        self.overwrite_results = overwrite_results
        self.logging_callbacks = logging_callbacks

    @property
    def name(self) -> str:
        return f'{self.mcmc_method()} MCMC'

    @abstractmethod
    def mcmc_method(self) -> MCMCMethod:
        raise NotImplementedError

    def __repr__(self) -> str:
        """A simple base representation. TODO overload this method in
        inheriting classes to make it more useful and specific to the method at
        hand.
        """
        return self.name

    def fpath(self, ident: str='') -> str:
        """Returns a file path to save the model to, based on its parameters."""
        return f'./results/mcmc/{self.mcmc_method()}_samples/{ident}.pt'

    def trainmodel(self, train_loader: DataLoader, ip: InferenceParams,
                   *args, **kwargs) -> None:
        raise AttributeError("MCMC doesn't support `trainmodel`")

    @abstractmethod
    def _do_sampling(self, p: Prospector, n_samples: Optional[int] = None) -> np.ndarray:
        """Actually run the MCMC sampling.

        Args:
            x: the input (photometric observations)
            n_samples: the number of samples to draw (from the end of the chain)

        Returns:
            np.ndarray: the `n_samples` samples.
        """
        raise NotImplementedError


    def sample(self, x: tensor_like, n_samples: Optional[int] = None,
               *args, **kwargs) -> Tensor:
        """Run MCMC sampling on the input x.

        Args:
            x: the input (photometric observations)
            n_samples: the number of samples to draw (from the end of the chain)

        Returns:
            Tensor: the `n_samples` samples.
        """

        # Since we need to retain the column names, we require either a structured
        # numpy array, or better still, the original pd.Series
        assert isinstance(x, pd.Series)
        galaxy = x

        p = Prospector(self.filters, self.emulate_ssp, galaxy)

        start_time = datetime.datetime.now()
        logging.info(f'Beginning MCMC ({self.mcmc_method()}) sample at {start_time}')

        denormed_samples = t.from_numpy(self._do_sampling(p, n_samples))
        norm_samples = normalise_theta(denormed_samples, cfg.FreeParams())

        duration = datetime.datetime.now() - start_time
        logging.info(f'{self.mcmc_method()} sampling took {duration.seconds} seconds')

        return norm_samples

    def save_samples(self, samples: Tensor, ident: str = ''):
        """Save some samples to disk.

        Args:
            samples: a NumPy array of samples.
            ident: some (unique) identifier (e.g. input row index, description
                of sapmling procedure etc.)
        """
        t.save(samples, self.fpath(ident))


# EMCEE -----------------------------------------------------------------------


class EMCEE(MCMC):

    def __init__(self, ep: EMCEEParams = cfg.EMCEEParams(),
                 overwrite_results: bool = False,
                 logging_callbacks: list[Callable] = []) -> None:
        super().__init__(ep, overwrite_results, logging_callbacks)
        self.ep = ep

    def mcmc_method(self) -> MCMCMethod:
        return types.EMCEE()

    def __repr__(self) -> str:
        return (f'EMCEE with {self.ep.nwalkers} walkers, running for '
                f'{self.ep.niter} iterations, with {self.ep.nburn} '
                f'burn in steps.')

    def _do_sampling(self, p: Prospector, n_samples: Optional[int] = None
                    ) -> np.ndarray:

        samples = self.ep.niter if n_samples is None else n_samples

        logging.info((
            f'Running EMCEE with {self.ep.nwalkers} walkers, {samples} '
            f'iterations, and {self.ep.nburn} burn in steps.'))

        # Parameters for prospector
        run_params: prun_params_t = p.run_params | {
                'emcee': True,
                'dynesty': False,
                'optimize': self.ep.optimize,
                'min_method': self.ep.min_method,
                'nwalkers': self.ep.nwalkers,
                'niter': samples,
                'nburn': self.ep.nburn,
                }

        output_dict = fit_model(p.obs, p.model, p.sps, lnprobfn=lnprobfn,
                                **run_params)
        sampling = output_dict['sampling']
        assert sampling[0] is not None
        sampler: EnsembleSampler = sampling[0]
        return sampler.flatchain


# Dynesty ----------------------------------------------------------------------


class Dynesty(MCMC):

    def __init__(self, dp: DynestyParams = cfg.DynestyParams(),
                 overwrite_results: bool = False,
                 logging_callbacks: list[Callable] = []) -> None:
        super().__init__(dp, overwrite_results, logging_callbacks)
        self.dp = dp

    def mcmc_method(self) -> MCMCMethod:
        return types.Dynesty()

    def __repr__(self) -> str:
        return (f'Dynesty (nested sampling)')

    def _do_sampling(self, p: Prospector, n_samples: Optional[int] = None
                    ) -> np.ndarray:
        logging.info('Running Dynesty (nested) sampling')

        samples = self.dp.maxcall if n_samples is None else n_samples

        run_params: prun_params_t = p.run_params | {
                'dynesty': True,
                'emcee': False,

                'nested_method': self.dp.method,
                'nlive_init': self.dp.nlive_init,
                'nlive_batch': self.dp.nlive_batch,
                'nested_dlogz_init': self.dp.dlogz_init,
                'nested_posterior_thresh': self.dp.posterior_thresh,
                'nested_maxcall': samples,
                'optimize': self.dp.optimize,
                'min_method': self.dp.min_method,
                'nmin': self.dp.nmin
        }

        output = fit_model(p.obs, p.model, p.sps, lnprobfn=lnprobfn,
                           **run_params)
        # TODO determine the type of 'sampling'.
        sampling = output['sampling'][0]
        assert sampling is not None
        return sampling.samples


# UltraNest --------------------------------------------------------------------


class UltraNest(MCMC):

    def __init__(self, up: UltraNestParams = cfg.UltraNestParams(),
                 overwrite_results: bool = False,
                 logging_callbacks: list[Callable] = []) -> None:
        super().__init__(up, overwrite_results, logging_callbacks)
        self.up = up

    def mcmc_method(self) -> MCMCMethod:
        return types.UltraNest()

    def _lnprobfn(self, model: SedModel, obs: cpz_obs_dict_t, sps: CSPSpecBasis,
                  nested: bool = True) -> Callable[[np.ndarray], Any]:
        """Closure wrapping Prospector's lnprobfn.
        The return type is either np.ndarray[floating[any]] or float. For
        simplicity, we just set it to `Any`.
        """
        def f(theta: np.ndarray) -> Any:
            p = lnprobfn(theta=theta, model=model, obs=obs, sps=sps, nested=nested)
            # TODO: Identify how frequently we are getting NaNs.
            # This probably happens when a dodgy parameter proposal is made...
            return np.nan_to_num(p)
        return f

    def _denormfn(self) -> Callable[[np.ndarray], np.ndarray]:
        lims = cfg.FreeParams()
        def f(theta: np.ndarray) -> np.ndarray:
            return denormalise_theta_np(theta, lims)
        return f

    def _do_sampling(self, p: Prospector, n_samples: Optional[int] = None
                    ) -> np.ndarray:

        logging.info('Running UltraNest sampling')

        samples = self.up.min_num_live_points if n_samples is None else n_samples

        sampler = ultranest.ReactiveNestedSampler(
                param_names = cfg.FreeParams().raw_members,
                loglike = self._lnprobfn(p.model, p.obs, p.sps),
                transform = self._denormfn(),
                ndraw_min = self.up.ndraw_min,
                ndraw_max = self.up.ndraw_max)

        result = sampler.run(
                    update_interval_volume_fraction = self.up.update_interval_volume_fraction,
                    show_status = self.up.show_status,
                    viz_callback = self.up.viz_callback,
                    dlogz = self.up.dlogz,
                    dKL = self.up.dKL,
                    frac_remain = self.up.frac_remain,
                    Lepsilon = self.up.Lepsilon,
                    min_ess = self.up.min_ess,
                    max_iters = self.up.max_iters,
                    max_ncalls = self.up.max_ncalls,
                    max_num_improvement_loops = self.up.max_num_improvement_loops,
                    min_num_live_points = samples,
                    cluster_num_live_points = self.up.cluster_num_live_points)

        # NOTE: we get lots of cool visualisations from the UltraNest sampler:
        # sampler.print_results()
        # sampler.plot_corner()
        # sampler.plot_run()
        # sampler.plot_trace()

        assert result is not None
        samples = result['samples']
        assert isinstance(samples, np.ndarray)
        return samples


def work_func(idx: int, x: pd.Series, survey: str):
    logging.info(f'Starting sampling for galaxy {idx}')

    ip = cfg.InferenceParams()
    logging.info(f'MCMC method is: {ip.mcmc_method}')
    mcmc: MCMC
    if ip.mcmc_method == types.EMCEE:
        mcmc = EMCEE(cfg.EMCEEParams())
    elif ip.mcmc_method == types.Dynesty:
        mcmc = Dynesty(cfg.DynestyParams())
    elif ip.mcmc_method == types.UltraNest:
        mcmc = UltraNest(cfg.UltraNestParams())
    else:
        raise ValueError('Unrecognized mcmc method.')

    samples = mcmc.sample(x)
    mcmc.save_samples(samples, f'{survey}_{idx}')
    logging.info(f'Saved samples for galaxy {idx}')


if __name__ == '__main__':
    cfg.configure_logging()

    ip = cfg.InferenceParams()
    survey = ip.filters.value  # use filter 'value' to identify survey

    # Draw samples from mcmcp.n_galaxies random galaxies from the survey.
    gs = sample_galaxies(ip.catalogue_loc, ip.filters, ip.mcmc_galaxies)
    args: list[tuple[int, pd.Series, str]] = \
        [(int(gs.index[i]), gs.iloc[i], survey) for i in range(len(gs))]

    with Pool(ip.mcmc_concurrency) as pool:
        pool.starmap(work_func, args)
    logging.info('Completed MCMC Sampling')

    # TODO generate mean and median parameter estimates for sampled galaxies

    # # optionally provide an index for a specific galaxy.
    # # index = 0
    # index = None

    # # catalogue: pd.DataFrame = load_catalogue(ip.catalogue_loc, Filters.DES)
    # galaxy, idx = load_galaxy(ip.catalogue_loc, Filters.DES, index)
    # samples = mcmc.sample(galaxy, 10000)
    # print(samples)
