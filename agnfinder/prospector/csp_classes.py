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

from prospect.sources import CSPSpecBasis
from agnfinder.fsps_emulation import emulate


class CSPSpecBasisAGN(CSPSpecBasis):
    """Override get_galaxy_spectrum to run as before, but add AGN component
    before returning.

    As with CSPSpecBasis, uses SSPBasis to implement get_spectrum(), which
    calls get_galaxy_spectrum and

    - applies observational effects
    - normalises by mass
    """

    def __init__(self,
            zcontinuous: int = 1,
            reserved_params: list[str] = ['zred', 'sigma_smooth'],
            vactoair_flag: bool = False,
            compute_vega_mags: bool = False,
            emulate_ssp: bool = False,
            **kwargs):
        # TODO remove kwargs if at all possible

        if emulate_ssp:
            # This is somewhat outdated but it serves as a good example of how
            # forward emulation works.
            logging.warning('Using custom FSPS emulator for SSP')
            self.ssp = CustomSSP()
        else:
            logging.warning('Using standard FSPS for SSP, no emulation')
            self.ssp = fsps.StellarPopulation(
                compute_vega_mags=compute_vega_mags,
                zcontinuous=zcontinuous,
                vactoair_flag=vactoair_flag)

        self.reserved_params = reserved_params

        self.params = {}

        # Make (unknown) kwargs properties of this class.
        # This is pretty bad form:
        # - No indication of types whatsoever
        # - Could result in attribute / class member conflicts
        # - Not safe
        # TODO remove if possible
        self.update(**kwargs)


        # TODO return here once quasar_templates is implemented
        self.quasar_template = quasar_templates.QuasarTemplate(
            template_loc=quasar_templates.INTERPOLATED_QUASAR_LOC
        )

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
        self.stellar_mass = None  # already exists
        self.params = CustomFSPSParams()  # dict of args to mimick FSPS
        self.careful = careful

    def relative_path(self, path: str) -> str:
        """Joins the provided path to the model base path.
        """
        return os.path.join(self.model_dir, path)

    def get_spectrum(self, tage: int) -> tuple[np.ndarray, np.ndarray]:
        if self.careful:
            assert tage is not 0
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

