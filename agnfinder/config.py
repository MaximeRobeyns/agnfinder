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
""" Project configuration file """

import math
import torch as t

from agnfinder.types import paramspace_t

# Configuration Values ========================================================
# Modify these!


# Keys prefixed by 'log_*' will be exponentiated later.
# TODO: perhaps save the spectra un-redshifted, then shift it at the end
free_params: paramspace_t = {
    'redshift': (0., 4.),
    # Mass of the galaxy
    'log_mass': (8, 12),
    # Brightness of the galaxy
    'log_agn_mass': (-7, math.log10(15)),  # from 10**-7 to 15
    # Scale of the agn torus
    'log_agn_torus_mass': (-7, math.log10(15)),
    'dust2': (0., 2.),
    'tage': (0.001, 13.8),  # bounds on this could be made tighter.
    'log_tau': (math.log10(.1), math.log10(30)),  # >2, has little effect
    'agn_eb_v': (0., 0.5),
    'inclination': (0., 90.)
}


# These defaults can be overridden by command line arguments when invoking
# agnfinder/simulation/simulation.py (run with --help flag to see options)
class SamplingParams():
    n_samples: int = 1000
    redshift_min: float = 0.
    redshift_max: float = 4.
    save_dir: str = './data'
    emulate_ssp: bool = False
    noise: bool = False
    filters: str = 'euclid'


# Utility classes -------------------------------------------------------------


class FreeParams():
    def __init__(self, params: paramspace_t):
        self.raw_params: paramspace_t = params

        self.params: t.Tensor = t.empty((0,2), dtype=t.float64)
        self.log: t.Tensor = t.empty((0,1), dtype=t.bool)

        for p in params:
            setattr(self, p, params[p])
            self.log = t.cat(
                (self.log, t.tensor([[1 if p.startswith('log') else 0]],
                                    dtype=t.bool)))
            self.params = t.cat(
                (self.params, t.tensor([params[p]], dtype=t.float64)))

        # Remove unnecessary singleton dimension in mask
        self.log = self.log.squeeze(-1)

    def __len__(self) -> int:
        return len(self.raw_params)
