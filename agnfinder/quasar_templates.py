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

import abc
import dill
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Callable
from scipy.integrate import simps
from scipy.interpolate import interp1d

import agnfinder.config as cfg


class InterpolatedTemplate(metaclass=abc.ABCMeta):

    def __init__(self, template_loc: str, data_loc: str = ""):
        self.data_loc: str = data_loc
        self.template_loc: str = template_loc

        if not self.data_loc == "":
            logging.warning('Creating new template - \
                    will overwrite existing templates on disk')
            self._interpolated_template = self._create_template()
            self._save_template()
        else:
            self._interpolated_template = self._load_template()

    @abc.abstractmethod
    def _create_template(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _eval_template(self):
        raise NotImplementedError

    def _save_template(self):
        with open(self.template_loc, 'wb') as f:
            dill.dump(self._interpolated_template, f)

    def _load_template(self):
        with open(self.template_loc, 'rb') as f:
            return dill.load(f)

    def __call__(self, *args, **kwargs):
        return self._eval_template(*args, **kwargs)


class QuasarTemplate(InterpolatedTemplate):

    def _eval_template(self, wavelengths: np.ndarray,
                       short_only: bool = False) -> np.ndarray:
        fluxes = 10**self._interpolated_template(np.log10(wavelengths))
        if short_only:
            # TODO implement this
            fluxes *= get_damping_multiplier(wavelengths, 'long')
        return fluxes

    def _create_template(self) -> np.ndarray:

        # radio-quiet mean quasar template from
        # https://iopscience.iop.org/article/10.1088/0067-0049/196/1/2#apjs400220f6.tar.gz

        df = pd.read_csv(self.data_loc, skiprows=19, delim_whitespace=True).read()
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
        normalised_interp = normalise_template(interp)
        return normalised_interp


class TorusModel():
    def __init__(self, model_loc: str):
        self.model_loc = model_loc
        self._interpolated_model = self.load_model()

    def load_model(self) -> np.ndarray:
        with open(self.model_loc, 'rb') as f:
            raw_interpolated_model = dill.load(f)
        # normalise at inclination=0, arbitrarily
        normalised_model: np.ndarray = normalise_template(raw_interpolated_model, 0)
        return normalised_model

    def __call__(self, wavelengths: np.ndarray, inclination: int,
                 long_only: bool = False) -> np.ndarray:
        fluxes = 10**self._interpolated_model(
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


def normalise_template(interp: interp1d, *extra_args):
    """# TODO (Maxime): descrbe what this does

    Args:
        interp: interpolation function

    Returns:
        A function to normalise the flux. Perhaps something like
        Callable[[np.ndarray, *], np.ndarray] but quite frankly I have no clue.
    """
    # in angstroms
    log_wavelengths = np.log10(np.logspace(np.log10(1e2), np.log10(1e7), 500000))

    total_flux = simps(10**interp(log_wavelengths, *extra_args),
                       10**log_wavelengths, dx=1, even='avg')

    # return normalised flux in log space (remembering that division is
    # subtraction)
    # Allegedly -21 so that agn mass is similar to galaxy mass.
    # This actually doesn't seem to be the case; is -21 arbitrary?
    return lambda x, *args: interp(x, *args) - np.log10(total_flux) - 21


if __name__ == '__main__':

    params = cfg.QuasarTemplateParams()

    # By providing data_loc we create a new template
    quasar = QuasarTemplate(params.interpolated_quasar_loc,
                            data_loc=params.quasar_data_loc)

    torus = TorusModel(params.torus_model_loc)

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
    plt.loglog(eval_wavelengths, quasar_only, labels='Quasar only')
    plt.loglog(eval_wavelengths, torus_only, label='Torus Only')
    plt.loglog(eval_wavelengths, net, label='Net')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (normalised)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(params.results_path('joint_quasar_torus_template.png'))
    plt.clf()

