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
"""Main Prospector problem generation class."""

import logging
import numpy as np
import pandas as pd
from typing import Callable, Optional

import agnfinder.config as cfg

from agnfinder.types import prun_params_t, FilterSet, SEDComponents
from agnfinder.config import CPzParams, SPSParams
from agnfinder.prospector import cpz_builders

class Prospector(object):

    def __init__(self, filter_selection: FilterSet, emulate_ssp: bool,
            galaxy: Optional[pd.Series] = None):
        """Construct a prospector class / 'problem' for simulation.

        Args:
            filter_selection: filters to use; e.g. Filters.Euclid.
            emulate_ssp: deprecated; always set this to false
            galaxy: an optional galaxy. If None, a dummy galaxy will be used to
                make prospector happy (e.g. for sampling).
        """
        logging.debug('initialising prospector class')

        # TODO try to remove these
        self.obs = cpz_builders.build_cpz_obs(filter_selection, galaxy)
        logging.debug(f'Created obs dict: {self.obs}')

        self.cpz_params = CPzParams()
        self.emulate_ssp = emulate_ssp
        self.model = cpz_builders.build_model(self.cpz_params)
        logging.debug(f'Created CPz model {self.model}')

        self.sps_params = SPSParams()
        self.sps = cpz_builders.build_sps(self.cpz_params, self.sps_params)
        logging.debug(f'Created sps model {self.sps}')

        self.run_params = self._cpz_params_to_run_params()
        logging.info(f'run params are: {self.run_params}')

    def calculate_sed(self, theta: Optional[np.ndarray] = None):
        self.model_spectra, self.model_photometry, _ = self.model.sed(
            theta = self.model.theta if theta is None else theta,
            obs=self.obs, sps=self.sps)

        # cosmological redshifting w_new = w_old * (1+z)
        a = 1.0 + self.model.params.get('zred', 0.)

        # redshift the *restframe* sps spectral wavelengths
        # wavelengths of source frame fluxes
        source_wavelengths = self.sps.wavelengths
        # redshift them via w_observed = w_source * (1+z); using z of model
        self.observer_wavelengths = source_wavelengths * a

    def _calculate_photometry(self, theta: np.ndarray) -> np.ndarray:
        """Private method, returning only the photometry resulting from the
        provided galaxy parameters.

        Args:
            theta: physical galaxy parameters

        Returns:
            np.ndarray: photometry
        """
        _, photometry, _ = self.model.sed(theta, obs=self.obs, sps=self.sps)
        return photometry

    def get_forward_model(self) -> Callable[[np.ndarray], np.ndarray]:
        """Closure to return a callable forward model, taking galaxy parameters
        (theta), and returning the corresponding photometry.

        Wraps the model, obs and sps properties from this class.

        Returns:
            Callable[[np.ndarray], np.ndarray]: The forward model function.
        """

        mass_idx = cfg.FreeParams().raw_members.index('log_mass')

        def f(theta: np.ndarray) -> np.ndarray:
            """Generate photometry from model parameters.

            Args:
                theta: The forward model parameters; must be denormalised.

            Returns:
                np.ndarray: photometry
            """
            # should be a single row of parameters
            if theta.ndim > 1:
                theta = theta.squeeze()
                assert theta.ndim == 1

            # Check mass is of correct order; >1e7.
            # (From the `raw_members` array in types.py:FreeParameters, the
            # mass parameter is at index 5)
            assert theta[mass_idx] > 1e7

            model_photometry = self._calculate_photometry(theta)

            return model_photometry

        return f

    def get_components(self) -> SEDComponents:
        """Returns SED components"""
        C = SEDComponents(
            wavelengths=self.sps.wavelengths,
            galaxy=self.sps.galaxy_flux,
            unextincted_quasar=self.sps.unextincted_quasar_flux,
            extincted_quasar=self.sps.extincted_quasar_flux,
            torus=self.sps.torus_flux,
            net=(self.sps.quasar_flux + self.sps.galaxy_flux))
        return C

    def _cpz_params_to_run_params(self) -> prun_params_t:
        """Casts the type-safe CPz Params (from config.py) to the parameter
        dictionary used by Prospector.

        Returns:
            prun_params_t: the parameter dictionary (e.g. passed to prospector classes)
        """

        run_params: prun_params_t = {}
        run_params['object_redshift'] = None
        run_params['fixed_metallicity'] = self.cpz_params.fixed_metallicity.value
        run_params['add_duste'] = True
        run_params['dust'] = True
        run_params['verbose'] = False
        run_params['zcontinuous'] = 1
        run_params['agn_mass'] = self.cpz_params.agn_mass.value
        run_params['agn_eb_v'] = self.cpz_params.agn_eb_v.value
        run_params['agn_torus_mass'] = self.cpz_params.agn_torus_mass.value
        run_params['igm_absorbtion'] = self.cpz_params.igm_absorbtion  # should this be bool?
        run_params['inclination'] = self.cpz_params.inclination.value
        run_params['emulate_ssp'] = self.emulate_ssp
        run_params['redshift'] = self.cpz_params.redshift.value

        return run_params
