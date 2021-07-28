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
from typing import Union, Callable

from agnfinder.config import CPzParams
from agnfinder.prospector import cpz_builders


class Prospector(object):

    run_params_t = tuple[str, Union[int, bool, float]]

    def __init__(self, filter_selection: str, emulate_ssp: bool):
        self.obs = cpz_builders.build_cpz_obs(filter_selection=filter_selection)
        logging.info(self.obs)

        cpz_params = CPzParams()

        self.model = cpz_builders.build_model(cpz_params)
        logging.info(self.model)

        self.sps = cpz_builders.build_sps(cpz_params, emulate_ssp, zcontinuous=1)
        logging.info(self.sps)

        self.run_params = self._cpz_params_to_run_params(cpz_params, emulate_ssp)

    def _cpz_params_to_run_params(self, params: CPzParams, emulate_ssp: bool
                                  ) -> run_params_t:
        """Casts the type-safe CPz Params (from config.py) to the parameter
        dictionary used by Prospector.

        Args:
            params: the CPz parameter object

        Returns:
            run_params_t: the parameter dictionary (e.g. passed to prospector classes)
        """
        run_params = {}
        run_params['object_redshift'] = None
        run_params['fixed_metallicity'] = params.fixed_metallicity.value
        run_params['add_duste'] = True
        run_params['dust'] = True
        run_params['verbose'] = False
        run_params['zcontinuous'] = 1
        run_params['agn_mass'] = params.agn_mass.value
        run_params['agn_eb_v'] = params.agn_eb_v.value
        run_params['agn_torus_mass'] = params.agn_torus_mass.value
        run_params['igm_absorbtion'] = params.igm_absorbtion  # is this always a bool?
        run_params['inclination'] = params.inclination.value
        run_params['emulate_ssp'] = emulate_ssp

        # TODO get redshift value before using this function
        run_params['redshift'] = None
        raise NotImplementedError

    def calculate_sed(self):
        self.model_spectra, self.model_photometry, _ = self.model.sed(
            self.model.theta, obs=self.obs, sps=self.sps)

        # cosmological redshifting w_new = w_old * (1+z)
        a = 1.0 + self.model.params.get('zred', 0.)

        # redshift the *restframe* sps spectral wavelengths
        # wavelengths of source frame fluxes
        source_wavelengths = self.sps.wavelengths
        # redshift them via w_observed = w_source * (1+z); using z of model
        self.observer_wavelengths = source_wavelengths * a


    def forward_model(self) -> Callable[[np.ndarray], np.ndarray]:
        # TODO determine what the output of the forward model is?
        # Are these numpy arrays, or floating point values
        def f(theta: np.ndarray) -> np.ndarray:
            """Generate photometry from model parameters.

            Args:
                theta: The forward model parameters; must be denormalised.

            Returns:
                np.ndarray: photometry
            """
            if theta.ndim > 1:
                theta = theta.squeeze()
                assert theta.ndim == 1

            # Check mass is properly large
            if 'zred' in model.free_params:
                mass_index = 1
            else:
                mass_index = 0
            asert theta[mass_index] > 1e7

            self.calculate_sed()
