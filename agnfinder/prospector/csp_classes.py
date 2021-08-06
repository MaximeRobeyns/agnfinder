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
"""Composite Stellar Population classes"""

import os
import fsps
import logging
import numpy as np

from typing import Any
from prospect.sources import CSPSpecBasis

import agnfinder.config as cfg
from agnfinder.fsps_emulation import emulate
from agnfinder import quasar_templates, extinction_models


class CSPSpecBasisAGN(CSPSpecBasis):
    """Override get_galaxy_spectrum to run as before, but add AGN component
    before returning.

    As with CSPSpecBasis, uses SSPBasis to implement get_spectrum(), which
    calls get_galaxy_spectrum and

    - applies observational effects
    - normalises by mass
    """

    # Note: it would be more elegant to pass in cpz_params, sps_params as args
    # to this constructor, but it must be consistent with constructor of super
    # class, hence the following:
    def __init__(self,
            zcontinuous: int = 1,
            reserved_params: list[str] = ['zred', 'sigma_smooth'],
            vactoair_flag: bool = False,
            compute_vega_mags: bool = False,
            emulate_ssp: bool = False,
            **kwargs):

        if emulate_ssp:
            # This is somewhat outdated but it serves as a good example of how
            # forward emulation would work.
            logging.warning('Using custom FSPS emulator for SSP')
            self.ssp = CustomSSP()
        else:
            logging.info('Using standard FSPS for SSP, no emulation')
            self.ssp = fsps.StellarPopulation(
                compute_vega_mags=compute_vega_mags,
                zcontinuous=zcontinuous,
                vactoair_flag=vactoair_flag)
            logging.debug('Successfully created fsps StellarPopulation model')

        self.reserved_params = reserved_params
        self.params: dict[str, Any] = {}
        self.update(**kwargs)

        quasar_params = cfg.QuasarTemplateParams()
        logging.info(f'quasar template parameters: {quasar_params}')

        self.quasar_template = quasar_templates.QuasarTemplate(
            template_loc=quasar_params.interpolated_quasar_loc,
            data_loc=quasar_params.quasar_data_loc,
            recreate_template=quasar_params.recreate_quasar_template
        )
        logging.debug(f'successfully initialised quasar template')

        self.torus_template = quasar_templates.TorusModel(
            quasar_params,
            template_loc=quasar_params.interpolated_torus_loc,
            data_loc=quasar_params.torus_data_loc,
            recreate_template=quasar_params.recreate_torus_template
        )
        logging.debug(f'successfully initialised torus model')

        extinction_params = cfg.ExtinctionTemplateParams()
        logging.info(f'extinction template parameters: {extinction_params}')
        self.extinction_template = extinction_models.ExtinctionTemplate(
            template_loc=extinction_params.interpolated_smc_extinction_loc,
            data_loc=extinction_params.smc_data_loc,
            recreate_template=extinction_params.recreate_extinction_template
        )
        logging.debug(f'successfully initialised extinction template')

        self.galaxy_flux = None
        self.unextincted_quasar_flux = None
        self.quasar_flux = None
        self.torus_flux = None
        self.extincted_quasar_flux = None

    def get_galaxy_spectrum(self, **params):
        """Update parameters, and then multiply SSP weights by SSP spectra and
        stellar masses, and then sum.
        """

        # This pattern is pretty bad, but it's what Prospector uses so we must
        # too...
        self.update(**params)

        try:
            self.params['agn_mass']
        except KeyError:
            raise AttributeError('Trying to calculate SED inc. AGN, but no \
                    agn_mass parameter is set')

        # no idea what this does (mfrac not accessed), but I'm leaving it in.
        mass = np.atleast_1d(self.params['mass']).copy()
        mfrac = np.zeros_like(mass)
        self.update_component(0)

        # Unfortunately we don't control this API:
        wave, spectrum = self.ssp.get_spectrum(
                tage=self.ssp.params['tage'])
        mfrac_sum = self.ssp.stellar_mass

        # rename to match
        mass_frac = mfrac_sum
        stellar_spectrum = spectrum

        # Insert blue AGN template here into spectrum
        # Normalised scale:
        template_quasar_flux = self.quasar_template(wave, short_only=True)
        quasar_flux = template_quasar_flux * self.params['agn_mass'] * 1e14

        # Normalised scale:
        template_torus_flux = self.torus_template(
                wave, self.params['inclination'], long_only=True)
        torus_flux = template_torus_flux * self.params['agn_torus_mass'] * 1e14

        # must always be specified, even if None
        # here float will be truthy; corresponds to OptionalValue(<something>)
        if self.params['agn_eb_v']:
            # This should go in the second monad callable
            extincted_quasar_flux = self.extinction_template(wave, quasar_flux, self.params['agn_eb_v'])
        else: # don't model
            extincted_quasar_flux = quasar_flux

        self.unextincted_quasar_flux = quasar_flux
        self.extincted_quasar_flux = extincted_quasar_flux
        self.torus_flux = torus_flux
        self.quasar_flux = extincted_quasar_flux + torus_flux
        self.galaxy_flux = stellar_spectrum

        # should be spectrum in Lsun/Hz per solar mass formed, restframe
        return wave, self.galaxy_flux + self.quasar_flux, mass_frac


class CSPSpecBasisNoEm(CSPSpecBasis):

    def get_galaxy_spectrum(self, **params):
        """Update parameters, then loop over each component getting a spectrum
        for each and sum with appropriate weights.

        Parameters:
            params: A parameter dictionary that gets passed to the self.update
                method, and will generally include physical parameters that
                control the stellar population and output spectrum or SED.

        Returns:
            wave: Wavelength in angstroms
            spectrum: Spectrum in units of Lsun/Hz/solar masses formed.
            mass_fraction: Fraction of the formed stellar mass that still
                 exists.
        """

        self.update(**params)

        spectra = []
        mass = np.atleast_1d(self.params['mass']).copy()
        mfrac = np.zeros_like(mass)

        # loop over mass components
        for i, _ in enumerate(mass):
            self.update_component(i)
            wave, spec = self.ssp.get_spectrum(tage=self.ssp.params['tage'],
                                               peraa=False)
            spectra.append(spec)
            mfrac[i] = (self.ssp.stellar_mass)

        # Convert normalisation units from per stellar mass to per mass formed
        if np.all(self.params.get('mass_units', 'mformed') == 'mstar'):
            mass /= mfrac
        spectrum = np.dot(mass, np.array(spectra)) / mass.sum()
        mfrac_sum = np.dot(mass, mfrac) / mass.sum()

        return wave, spectrum, mfrac_sum


class CustomSSP():
    """Replicates fsps.StellarPopulation"""

    def __init__(self, careful: bool = True, model_dir: str = 'data'):
        logging.warning('Using cached SSP.')
        self.model_dir = model_dir

        num_params = 3
        num_bases = 10
        gp_model_loc = self.relative_path(f'gpfit_{num_bases}_{num_params}.zip')
        pca_model_loc = self.relative_path('pcaModel.pickle')
        self._spectrum_emulator = emulate.GPEmulator(
            gp_model_loc=gp_model_loc,
            pca_model_loc=pca_model_loc
        )

        mass_model_loc = self.relative_path('mass_emulator.pickle')
        self._mass_emulator = emulate.SKLearnEmulator(
            model_loc=mass_model_loc
        )

        reference_wave_loc = self.relative_path('reference_wave.txt')
        self.wavelengths = np.loadtxt(reference_wave_loc)

        # This this is unfortunate, but unfortunately we don't control the API
        self.stellar_mass: Any = None  # already exists
        self.params = CustomFSPSParams()  # dict of args to mimick FSPS
        self.careful = careful

    def relative_path(self, path: str) -> str:
        """Joins the provided path to the model base path.
        """
        return os.path.join(self.model_dir, path)

    def get_spectrum(self, tage: int) -> tuple[np.ndarray, np.ndarray]:
        if self.careful:
            assert tage != 0
            self.check_fixed_params_unchanged()
        param_vector = np.array([
            self.params['tau'], tage, self.params['dust2']])

        # emulator doesn't model the first 100 wavelengths (which are ~0)
        # because of how Nesar made it. Add then back manually
        spectra = np.hstack([
            np.ones(100) * 1e-60, self._spectrum_emulator(param_vector)])
        self.stellar_mass = self._mass_emulator(param_vector)
        return self.wavelengths, spectra

    def check_fixed_params_unchanged(self):
        """Does what it says on the tin.

        Raises:
            ValueError: If one of the fixed parameters was changed.
        """
        expected_fixed_args = {
            'logzsol': 0.0,
            'sfh': 4,
            'imf_type': 2,
            'dust_type': 2,
            'add_dust_emission': True,
            'duste_umin': 1.0,
            'duste_qpah': 4.0,
            'duste_gamma': 0.001,
            'add_igm_absorption': True,
            'igm_factor': 1.0
        }
        for k, v in expected_fixed_args.items():
            if not self.params[k] == v:
                raise ValueError(
                    "Expected value {} for fixed parameter {} but got {}"
                    .format(self.params[k], k, v))


class CustomFSPSParams():

    def __init__(self):

        self.ssp_params: list[str] = ["imf_type", "imf_upper_limit",
                "imf_lower_limit", "imf1", "imf2", "imf3", "vdmc", "mdave",
                "dell", "delt", "sbss", "fbhb", "pagb", "add_stellar_remnants",
                "tpagb_norm_type", "add_agb_dust_model", "agb_dust", "redgb",
                "agb", "masscut", "fcstar", "evtype", "smooth_lsf"]

        self.csp_params: list[str] = ["smooth_velocity", "redshift_colors",
                "compute_light_ages","nebemlineinspec", "dust_type",
                "add_dust_emission", "add_neb_emission", "add_neb_continuum",
                "cloudy_dust", "add_igm_absorption", "zmet", "sfh", "wgp1",
                "wgp2", "wgp3", "tau", "const", "tage", "fburst", "tburst",
                "dust1", "dust2", "logzsol", "zred", "pmetals", "dust_clumps",
                "frac_nodust", "dust_index", "dust_tesc", "frac_obrun", "uvb",
                "mwr", "dust1_index", "sf_start", "sf_trunc", "sf_slope",
                "duste_gamma", "duste_umin", "duste_qpah", "sigma_smooth",
                "min_wave_smooth", "max_wave_smooth", "gas_logu", "gas_logz",
                "igm_factor", "fagn", "agn_tau"]

        self._params = {}

    @property
    def all_params(self) -> list[str]:
        return self.ssp_params + self.csp_params

    def __getitem__(self, k: str):
        return self._params[k]

    def __setitem__(self, k: str, v):
        # TODO no clue what types v will be...
        self._params[k] = v
