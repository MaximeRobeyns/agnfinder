# AGNfinder: Detect AGN from photometry in XXL data.
#
# Copyright (C) 2021 Maxime Robeyns <maximerobeyns@gmail.com>
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
"""Tests for various prospector utility functions"""


from agnfinder.config import CPzParams
from agnfinder.types import prun_params_t
from agnfinder.prospector import Prospector

class ProspectorTest(Prospector):
    def __init__(self, filter_selection: str, emulate_ssp: bool):
        self.filter_selection = filter_selection
        self.emulate_ssp = emulate_ssp
        self.cpz_params = CPzParams()

def test_cpz_params_to_run_params():

    p = ProspectorTest('euclid', True)

    # Verify that TestProspector has correctly inherited all the Prospector
    # methods
    methods = ['calculate_sed', '_calculate_photometry', 'get_forward_model',
               '_cpz_params_to_run_params']
    for m in methods:
        op = getattr(p, m, None)
        assert callable(op)

    rp: prun_params_t = p._cpz_params_to_run_params()

    # These are the same parameters used to initialise the TestProspector
    # class (defined in config.py).
    cp = CPzParams()

    # fixed CPz Parameters:
    assert rp['object_redshift'] is None
    assert rp['add_duste']
    assert rp['dust']
    assert not rp['verbose']
    assert rp['zcontinuous'] == 1

    # Configuration-based CPz parameters
    assert rp['emulate_ssp'] == p.emulate_ssp
    assert rp['redshift'] == cp.redshift.value
    assert rp['fixed_metallicity'] == cp.fixed_metallicity.value
    assert rp['agn_mass'] == cp.agn_mass.value
    assert rp['agn_eb_v'] == cp.agn_eb_v.value
    assert rp['agn_torus_mass'] == cp.agn_torus_mass.value
    assert rp['igm_absorbtion'] == cp.igm_absorbtion  # is this always a bool?
    assert rp['inclination'] == cp.inclination.value

