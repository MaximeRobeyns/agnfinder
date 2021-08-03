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
"""Defines ExtinctionTemplate"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from agnfinder import quasar_templates
import agnfinder.config as cfg


class ExtinctionTemplate(quasar_templates.InterpolatedTemplate):

    def _create_template(self) -> interp1d:
        df = pd.read_csv(self.data_loc, delim_whitespace=True)
        assert isinstance(df, pd.DataFrame)
        return interp1d(df['wavelength'], df['k_l'], kind='linear',
                        fill_value=0, bounds_error=False)

    def __call__(self, wavelength: np.ndarray, flux: np.ndarray,
                 eb_v: float) -> np.ndarray:
        return flux * \
            10**(-0.4 * self._interpolated_template(wavelength, None) * eb_v)


if __name__ == '__main__':

    params = cfg.ExtinctionTemplateParams()

    smc_extinction = ExtinctionTemplate(
        template_loc=params.interpolated_smc_extinction_loc,
        data_loc=params.smc_data_loc,
        recreate_template=True
    )
    eb_v_values = list(np.linspace(0.1, 0.5, 5))
    eval_wavelengths = np.logspace(
            np.log10(100), np.log10(40000), 5000)  # in angstroms
    uniform_flux = np.ones_like(eval_wavelengths)

    # SMC Extinction plot -----------------------------------------------------

    plt.loglog(eval_wavelengths, uniform_flux, 'k--', label='Initial')
    for eb_v in eb_v_values:
        plt.loglog(eval_wavelengths,
                   smc_extinction(eval_wavelengths, uniform_flux, eb_v),
                   label=f'Extincted (EB_V={eb_v:.1f})')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux')
    plt.legend()
    plt.tight_layout()
    plt.savefig(params.results_path('smc_extinction.png'))
