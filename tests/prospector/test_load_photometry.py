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
"""Tests filter loading"""

import pytest
import collections
import pandas as pd

from sedpy import observate

from agnfinder.types import Filters
from agnfinder.prospector import load_photometry

def test_Filter():
    fs = [
        load_photometry.Filter(
            bandpass_file=f'{b}_galex',
            mag_col=f'mag_auto_galex_{b.lower()}_dr67',
            error_col=f'magerr_auto_galex_{b.lower()}_dr67')
        for b in ['NUV', 'FUV']]

    f1: load_photometry.Filter = fs[0]
    assert f1.bandpass_file == 'NUV_galex'
    assert f1.mag_col == 'mag_auto_galex_nuv_dr67'
    assert f1.mag_error_col == 'magerr_auto_galex_nuv_dr67'
    assert f1.maggie_col == 'maggie_auto_galex_nuv_dr67'
    assert f1.maggie_error_col == 'maggieerr_auto_galex_nuv_dr67'

    f2: load_photometry.Filter = fs[1]
    assert f2.bandpass_file == 'FUV_galex'
    assert f2.mag_col == 'mag_auto_galex_fuv_dr67'
    assert f2.mag_error_col == 'magerr_auto_galex_fuv_dr67'
    assert f2.maggie_col == 'maggie_auto_galex_fuv_dr67'
    assert f2.maggie_error_col == 'maggieerr_auto_galex_fuv_dr67'

def test_get_filters():

    with pytest.raises(ValueError):
        _ = load_photometry.get_filters('asdf')

    fs_reliable = load_photometry.get_filters('reliable')
    reliable_filters = [ f'{f}_sloan' for f in ['u', 'g', 'r', 'i', 'z']] + \
                       [ f'VISTA_{f}' for f in ['H', 'J', 'Ks', 'Y', 'Z']] + \
                       [ f'wise_{f}' for f in ['w1', 'w2']]
    bp_reliable = [f.bandpass_file for f in fs_reliable]
    assert collections.Counter(bp_reliable) == \
        collections.Counter(reliable_filters)

    fs_euclid = load_photometry.get_filters('euclid')
    euclid_filters = [ f'{f}_sloan' for f in ['u', 'g', 'r', 'i', 'z']] + \
                     [ f'VISTA_{f}' for f in ['H', 'J', 'Y']]
    bp_euclid = [f.bandpass_file for f in fs_euclid]
    assert collections.Counter(bp_euclid) == \
        collections.Counter(euclid_filters)

    fs_all = load_photometry.get_filters('all')
    all_filters = reliable_filters + \
                  [ f'{f}_galex' for f in ['NUV', 'FUV']] + \
                  [ f'{f}_cfhtl' for f in ['g', 'r', 'u', 'z']] + \
                  [ 'i_cfhtl_new' ] + \
                  [ f'{f}_kids' for f in ['i', 'r'] ]
    bp_all = [f.bandpass_file for f in fs_all]
    assert collections.Counter(bp_all) == \
        collections.Counter(all_filters)


def test_load_galaxy():
    catalogue_loc = './data/cpz_paper_sample_week3.parquet'
    galaxy, _ = load_photometry.load_galaxy(catalogue_loc)
    assert(isinstance(galaxy, pd.Series))


# TODO test loading from .fits not just .parquet
def test_load_galaxy_for_prospector():
    catalogue_loc = './data/cpz_paper_sample_week3.parquet'
    galaxy, _ = load_photometry.load_galaxy(catalogue_loc)
    filters, _, _ = load_photometry.load_galaxy_for_prospector(
        galaxy, filter_selection=Filters.Euclid)
    assert all([isinstance(f, observate.Filter) for f in filters])


def test_load_dummy_galaxy():
    filters, _, _ = load_photometry.load_dummy_galaxy('euclid')
    assert all([isinstance(f, observate.Filter) for f in filters])

    filters, _, _ = load_photometry.load_dummy_galaxy('reliable')
    assert all([isinstance(f, observate.Filter) for f in filters])

    filters, _, _ = load_photometry.load_dummy_galaxy('all')
    assert all([isinstance(f, observate.Filter) for f in filters])
