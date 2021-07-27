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

import logging

from prospect.sources import CSPSpecBasis


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
            logging.warning('Using custom FSPS emulator for SSP')
            self.ssp = CustomSSP()
            # TODO return here after CustomSSP is implemented


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

