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
"""Create and load quasar templates."""

import os
import abc
import dill
import h5py
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Union, Callable, Optional
from scipy.integrate import simps
from scipy.interpolate import interp1d, interp2d

import agnfinder.config as cfg


class InterpolatedTemplate(metaclass=abc.ABCMeta):
    """This class is inherited by both QuasarTemplate and ExtinctionTemplate"""

    def __init__(self, template_loc: str, data_loc: str = "",
                 recreate_template: bool = False):
        """Loads a model for some quasar object.

        Args:
            template_loc: Location on disk of created model.
            data_loc: Location of data used to create template.
            recreate_template: Recreate template from data.

        Raises:
            RuntimeError: If template cannot be loaded, or data_loc unspecified
                when recreating template.
        """
        self.data_loc: str = data_loc
        self.template_loc: str = template_loc

        self._load_error = False

        if not recreate_template:
            try:
                logging.debug(f'Opening quasar template: {self.template_loc}')
                self._interpolated_template = self._load_template()
            except Exception as e:
                logging.error(f'Error opening template: {e}')
                self._load_error = True

        if recreate_template or self._load_error:
            if data_loc == "":
                err = f'No data location specified for template {template_loc}'
                logging.error(err)
                raise RuntimeError(err)
            if not os.path.exists(data_loc):
                raise ValueError(f'Data location {data_loc} does not exist')
            logging.warning(
                f'Creating new template {template_loc} from data {data_loc}')
            self._interpolated_template = self._create_template()
            self._save_template()

    @abc.abstractmethod
    def _create_template(self) -> Callable[[np.ndarray], np.ndarray]:
        """Create the template from data"""
        raise NotImplementedError

    # Need to think a bit more carefully about the type of this function...
    # @abc.abstractmethod
    # def __call__(self, wavelengths: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
    #     """Evalaute the template for some wavelengths (in angstroms), returning
    #     fluxes.

    #     Note: arguments and returned values are _not_ in log space.
    #     """
    #     raise NotImplementedError

    def _save_template(self):
        """Saves a newly created template to disk for faster loading later"""
        with open(self.template_loc, 'wb') as f:
            dill.dump(self._interpolated_template, f)

    def _load_template(self) -> Callable[[np.ndarray], np.ndarray]:
        """Loads an existing template form disk"""
        with open(self.template_loc, 'rb') as f:
            return dill.load(f)


    def normalise_template(self, interp: interp1d, *extra_args
                          ) -> Callable[[np.ndarray], np.ndarray]:
        """Normalises the template mapping wavelengths to flux, such that it
        returns the normalised flux in log space.

        Args:
            interp: the loaded model

        Returns:
            A function to normalise the flux. Perhaps something like
            Callable[[np.ndarray, *], np.ndarray] but quite frankly I have no clue.
        """
        # in angstroms
        log_wavelengths = np.log10(
            np.logspace(np.log10(1e2), np.log10(1e7), 500000))

        total_flux = simps(10**interp(log_wavelengths, *extra_args),
                           10**log_wavelengths, dx=1, even='avg')

        # Return normalised flux in log space (remembering that division is
        # subtraction)
        #
        # -21 is a magic constant :(
        #
        # Allegedly -21 is so that agn mass is similar to galaxy mass, but this
        # actually doesn't seem to be the case; is this arbitrary?
        return lambda x: interp(x) - np.log10(total_flux) - 21


class QuasarTemplate(InterpolatedTemplate):

    def _create_template(self) -> Callable[[np.ndarray], np.ndarray]:

        # radio-quiet mean quasar template from
        # https://iopscience.iop.org/article/10.1088/0067-0049/196/1/2#apjs400220f6.tar.gz

        df = pd.read_csv(self.data_loc, skiprows=19, delim_whitespace=True)
        assert isinstance(df, pd.DataFrame)
        log_freq = df['log_freq']
        assert log_freq is not None
        freqs = 10**log_freq.to_numpy()

        wavelengths = 299792458. / freqs  # 'Research code' ladies and gentlemen!

        # linear interpolation in log wavelength(A)/log freq(Hz) space
        interp = interp1d(
            np.log10(wavelengths * 1e10),  # in angstroms
            df['log_flux'],
            kind='linear'
        )
        normalised_interp = self.normalise_template(interp)
        return normalised_interp

    def __call__(self, wavelengths: np.ndarray,
                       short_only: bool = False) -> np.ndarray:
        fluxes = 10**self._interpolated_template(np.log10(wavelengths))
        if short_only:
            fluxes *= get_damping_multiplier(wavelengths, 'long')
        return fluxes


class TorusModel():
    """This is like InterpolateTemplate2d, but since only one class would
    inherit it, we simply define the TorusModel class on its own.
    """

    def __init__(self, params: cfg.QuasarTemplateParams, template_loc: str,
                 data_loc: str = "", recreate_template: bool = False):
        self._params = params
        self.data_loc: str = data_loc
        self.template_loc: str = template_loc
        self._load_error = False

        if not recreate_template:
            try:
                logging.debug(f'Opening quasar template: {self.template_loc}')
                self._interpolated_template = self._load_template()
            except Exception as e:
                logging.error(f'Error opening template: {e}')
                self._load_error = True

        if recreate_template or self._load_error:
            if data_loc == "":
                err = f'No data location specified for template {template_loc}'
                logging.error(err)
                raise RuntimeError(err)
            if not os.path.exists(data_loc):
                raise ValueError(f'Data location {data_loc} does not exist')
            logging.warning(
                f'Creating new template {template_loc} from data {data_loc}')
            self._interpolated_template = self._create_template()
            self._save_template()

    def _save_template(self):
        with open(self.template_loc, 'wb') as f:
            dill.dump(self._interpolated_template, f)

    def _create_template(self) -> Callable[[np.ndarray, int], np.ndarray]:

        # Data from, saved to self.data_loc
        # https://www.clumpy.org/pages/seds.html
        # Also see https://iopscience.iop.org/article/10.1086/590483/pdf
        # (https://arxiv.org/abs/0806.0511)

        with h5py.File(self.data_loc, 'r') as f:
            keys = ['wave', 'sig', 'i', 'N0', 'q', 'Y', 'tv', 'flux_toragn']
            assert all([k in f for k in keys])
            dsets: list[h5py.Dataset] = [f[k] for k in keys]
            arrs: list[np.ndarray] = [d[...] for d in dsets]

            wavelengths = arrs[0] * 1e4
            opening_angle = arrs[1]
            inclination = arrs[2]
            n0 = arrs[3]
            q = arrs[4]
            y = arrs[5]
            tv = arrs[6]
            seds = arrs[7]

        suggested_fixed_params = \
            (n0 == self._params.torus_n0) & \
            (opening_angle == self._params.torus_opening_angle) & \
            (q == self._params.torus_q) & \
            (y == self._params.torus_y) & \
            (tv == self._params.torus_tv)

        func = interp2d(x=np.log10(wavelengths),
               y=inclination[suggested_fixed_params],
               z=np.log10(seds[suggested_fixed_params]))

        # We normalise at inclination=0, arbitrarily
        return self.normalise_template(func, 0)

    def _load_template(self) -> Callable[[np.ndarray, int], np.ndarray]:
        with open(self.template_loc, 'rb') as f:
            return dill.load(f)

    def normalise_template(self, interp: interp2d, *extra_args
                          ) -> Callable[[np.ndarray, int], np.ndarray]:
        # in angstroms
        log_wavelengths = np.log10(
            np.logspace(np.log10(1e2), np.log10(1e7), 500000))

        total_flux = simps(10**interp(log_wavelengths, *extra_args),
                           10**log_wavelengths, dx=1, even='avg')

        return lambda x, y: interp(x, y) - np.log10(total_flux) - 21

    def __call__(self, wavelengths: np.ndarray, inclination: int,
                 long_only: bool = False) -> np.ndarray:

        fluxes = 10**self._interpolated_template(
                np.log10(wavelengths), inclination)

        if long_only:  # add exponential damping after 1 micron
            fluxes *= get_damping_multiplier(wavelengths, 'short')

        return fluxes


def get_damping_multiplier(wavelengths: np.ndarray, damp: str) -> np.ndarray:
    damping_multiplier = np.ones_like(wavelengths)
    if damp == 'long':  # damp wavelengths above 1 micron
        to_damp = wavelengths > 1e4
        log_m = -5
    elif damp == 'short':
        to_damp = wavelengths < 1e4
        log_m = 5
    else:
        raise ValueError(f'damp={damp} not understood')
    intercept = 1e4 ** (-1 * log_m)
    damping_multiplier[to_damp] = intercept * wavelengths[to_damp] ** log_m
    return damping_multiplier



if __name__ == '__main__':

    params = cfg.QuasarTemplateParams()

    # By providing data_loc we create a new template
    quasar = QuasarTemplate(
        template_loc=params.interpolated_quasar_loc,
        data_loc=params.quasar_data_loc,
        recreate_template=True)

    torus = TorusModel(
        params,
        template_loc=params.interpolated_torus_loc,
        data_loc=params.torus_data_loc,
        recreate_template=True)

    eval_wavelengths = np.logspace(
            np.log10(1e2), np.log10(1e8), 500000)  # in angstroms

    # Damping multiplier plot -------------------------------------------------

    logging.info('Creating damping multiplier plot')
    plt.loglog(eval_wavelengths,
               get_damping_multiplier(eval_wavelengths, 'long'),
               label='long')
    plt.loglog(eval_wavelengths,
               get_damping_multiplier(eval_wavelengths, 'short'),
               label='short')
    plt.legend()
    plt.savefig(params.results_path('damping_multiplier.png'))
    plt.clf()  # clear the current figure

    # Quasar template plot ----------------------------------------------------

    logging.info('Creating quasar template plot')
    plt.loglog(eval_wavelengths, quasar(eval_wavelengths), label='Original')
    plt.loglog(eval_wavelengths, quasar(eval_wavelengths, short_only=True),
               label='Without dust')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (normalised)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(params.results_path('quasar_template.png'))
    plt.clf()

    # Torus template plot -----------------------------------------------------

    logging.info('Creating torus template plot')
    plt.loglog(eval_wavelengths, torus(eval_wavelengths, inclination=10),
               label='Original')
    plt.loglog(eval_wavelengths,
               torus(eval_wavelengths, inclination=10, long_only=True),
               label='Without blue')
    plt.loglog(eval_wavelengths, torus(eval_wavelengths, inclination=45),
               label='Inc=45')
    plt.loglog(eval_wavelengths, torus(eval_wavelengths, inclination=90),
               label='Inc=90')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux  (normalised)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(params.results_path('torus_template.png'))
    plt.clf()

    # Joint quasar torus template plot -----------------------------------------

    quasar_only = quasar(eval_wavelengths, short_only=True)
    torus_only = torus(eval_wavelengths, inclination=80, long_only=True)
    net = quasar_only + torus_only
    plt.loglog(eval_wavelengths, quasar_only, label='Quasar only')
    plt.loglog(eval_wavelengths, torus_only, label='Torus Only')
    plt.loglog(eval_wavelengths, net, label='Net')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (normalised)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(params.results_path('joint_quasar_torus_template.png'))
    plt.clf()

