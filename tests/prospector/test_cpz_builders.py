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
"""Tests builders for 'classification-aided photometric redshfit' estimation
module"""


import pytest
from prospect.models import priors, templates

from agnfinder.prospector import cpz_builders
from agnfinder.types import pdict_t, \
        MaybeFloat, Free, Just, \
        Optional, OptionalValue, Nothing


def test_maybe_monad():
    params = {}

    # Free --------------------------------------------------------------------
    agn_mass: MaybeFloat = Free
    with pytest.raises(KeyError):
        params['agn_mass'] |= agn_mass.use(
                cpz_builders._use_free('agn mass', {'N': 1, 'init': 1,
                          'prior': priors.LogUniform(mini=1e-7, maxi=15)}),
                cpz_builders._use_float('agn_mass', {'N': 1}))
    params['agn_mass'] = agn_mass.use(
            cpz_builders._use_free('agn mass', {'N': 1, 'init': 1,
                      'prior': priors.LogUniform(mini=1e-7, maxi=15)}),
            cpz_builders._use_float('agn_mass', {'N': 1}))
    assert params['agn_mass'] is not None
    pam: pdict_t = params['agn_mass']
    assert pam['isfree']
    assert pam['N'] == 1
    assert pam['init'] == 1
    assert isinstance(pam['prior'], priors.LogUniform)

    assert agn_mass.value == True

    # Just(<float>) ----------------------------------------------------------
    agn_mass = Just(0.5)
    params['agn_mass'] = agn_mass.use(
            cpz_builders._use_free('agn mass', {'N': 1, 'init': 1,
                      'prior': priors.LogUniform(mini=1e-7, maxi=15)}),
            cpz_builders._use_float('agn_mass', {'N': 1}))
    assert params['agn_mass'] is not None
    pam: pdict_t = params['agn_mass']
    assert not pam['isfree']
    assert pam['N'] == 1
    assert pam['init'] == 0.5

    assert agn_mass.value == 0.5


def test_optional_monad():

    params = {}

    # Nothing -----------------------------------------------------------------
    agn_eb_v: Optional = Nothing
    params['agn_eb_v'] = agn_eb_v.use(
            cpz_builders._use_nothing('AGN disk'),
            cpz_builders._use_free('agn_eb_v', {'N': 1, 'init': 0.1,
                    'units': '', 'prior': priors.TopHat(mini=0., maxi=0.5)}),
            cpz_builders._use_float('agn_eb_v', {'N': 1, 'units': '',
                    'prior': priors.TopHat(mini=0., maxi=0.5)}))
    assert params['agn_eb_v'] is None
    assert agn_eb_v.value == False

    # OptionalValue(Free) -----------------------------------------------------
    agn_eb_v = OptionalValue(Free)
    params['agn_eb_v'] = agn_eb_v.use(
            cpz_builders._use_nothing('AGN disk'),
            cpz_builders._use_free('agn_eb_v', {'N': 1, 'init': 0.1,
                    'units': '', 'prior': priors.TopHat(mini=0., maxi=0.5)}),
            cpz_builders._use_float('agn_eb_v', {'N': 1, 'units': '',
                    'prior': priors.TopHat(mini=0., maxi=0.5)}))
    assert params['agn_eb_v'] is not None
    pebv: pdict_t = params['agn_eb_v']
    assert pebv['isfree']
    assert pebv['N'] == 1
    assert pebv['init'] == 0.1
    assert pebv['units'] == ''
    assert isinstance(pebv['prior'], priors.TopHat)
    assert agn_eb_v.value == True

    # OptionalValue(Just(<float>)) --------------------------------------------
    agn_eb_v = OptionalValue(Just(0.5))
    params['agn_eb_v'] = agn_eb_v.use(
            cpz_builders._use_nothing('AGN disk'),
            cpz_builders._use_free('agn_eb_v', {'N': 1, 'init': 0.1,
                    'units': '', 'prior': priors.TopHat(mini=0., maxi=0.5)}),
            cpz_builders._use_float('agn_eb_v', {'N': 1, 'units': '',
                    'prior': priors.TopHat(mini=0., maxi=0.5)}))
    assert params['agn_eb_v'] is not None
    pebv: pdict_t = params['agn_eb_v']
    assert not pebv['isfree']
    assert pebv['N'] == 1
    assert pebv['units'] == ''
    assert pebv['init'] == 0.5
    assert isinstance(pebv['prior'], priors.TopHat)
    assert agn_eb_v.value == 0.5


def test_known_keys():

    model_params = templates.TemplateLibrary['parametric_sfh']
    model_params |= templates.TemplateLibrary['dust_emission']
    model_params |= templates.TemplateLibrary['igm']

    keys = model_params.keys()

    known_keys = (['logzsol', 'zred'])
    unknown_keys = (['agn_mass', 'agn_eb_v', 'agn_torus_mass', 'inclination'])

    assert all(k in keys for k in known_keys)
    assert all(not k in keys for k in unknown_keys)


