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
"""Loads filters and other data."""

import random
import logging
import numpy as np
import pandas as pd
import torch as t

from sedpy import observate
from typing import Optional, Union
from astropy.io import fits

from . import columns

from agnfinder.types import FilterSet, Filters


class Filter(object):
    def __init__(self, bandpass_file: str, mag_col: str, error_col: str):
        """Encapsulates a sedpy filter.

        Args:
            bandpass_file: Location of the bandpass_file
            mag_col: Name of the maggies column in the dataframe.
            error_col: Name of the errors column.
        """
        self.bandpass_file = bandpass_file
        self.mag_col = mag_col
        self.mag_error_col = error_col

        self.maggie_col = mag_col.replace('mag', 'maggie')
        self.maggie_error_col = error_col.replace('mag', 'maggie')


def get_filters(filter_selection: FilterSet) -> list[Filter]:
    """Get the list of Filter objects corresponding to the named filter selection.

    Args:
        filter_selection: The filter selection; Filters.{Reliable, Euclid, All}

    Returns:
        list[Filter]: A list all the filters for the selection.

    Raises:
        ValueError: If the filter selection is not recognised.
    """

    galex = [
        Filter(
            bandpass_file=f'{b}_galex',
            mag_col=f'mag_auto_galex_{b.lower()}_dr67',
            error_col=f'magerr_auto_galex_{b.lower()}_dr67')
        for b in ['NUV', 'FUV']]

    # cfht is awkward due to i filter renaming. For now, we use i = i_new
    cfht = [
        Filter(
            bandpass_file='{}_cfhtl{}'.format(
                b, '_new' if b == 'i' else ''
            ),
            mag_col=f'mag_auto_cfhtwide_{b}_dr7',
            error_col=f'magerr_auto_cfhtwide_{b}_dr7')
        for b in ['g', 'i', 'r', 'u', 'z']]

    des = [
        Filter(
            bandpass_file=f'DES_{b}',
            mag_col=f'mag_auto_{b}',
            error_col=f'magerr_auto_{b}')
        for b in ['g', 'i', 'r']]

    kids = [
        Filter(
            bandpass_file=f'{b}_kids',
            mag_col=f'mag_auto_kids_{b}_dr2',
            error_col=f'magerr_auto_kids_{b}_dr2')
        for b in ['i', 'r']]

    vista = [
        Filter(
            bandpass_file=f'VISTA_{b}',
            mag_col='mag_auto_viking_{}_dr2'.format(b.lower().strip('s')),
            error_col='magerr_auto_viking_{}_dr2'.format(b.lower().strip('s')))
        for b in ['H', 'J', 'Ks', 'Y', 'Z']]

    vista_euclid = [
        Filter(
            bandpass_file=f'VISTA_{b}',
            mag_col='mag_auto_viking_{}_dr2'.format(b.lower().strip('s')),
            error_col='magerr_auto_viking_{}_dr2'.format(b.lower().strip('s')))
        for b in ['H', 'J', 'Y']]

    vista_des = [
        Filter(
            bandpass_file=f'VISTA_{b}',
            mag_col=f'{b}AUTOMAG',
            error_col=f'{b}AUTOMAGERR')
        for b in ['H', 'J', 'Y', 'Z']]

    sdss = [
        Filter(
            bandpass_file=f'{b}_sloan',
            mag_col=f'mag_auto_sdss_{b}_dr12',
            error_col=f'magerr_auto_sdss_{b}_dr12')
        for b in ['u', 'g', 'r', 'i', 'z']]

    wise = [
        Filter(
            bandpass_file=f'wise_{b}',
            mag_col='mag_auto_AllWISE_{b.upper()}',
            error_col=f'magerr_auto_AllWISE_{b.upper()}')
        for b in ['w1', 'w2']]  # exclude w3, w4

    # These are _not_ in wavelength order.
    all_filters = galex + sdss + cfht + kids + vista + wise

    if filter_selection == 'des':
        return des + vista_des
    elif filter_selection == 'reliable':
        return sdss + vista + wise
    elif filter_selection == 'euclid':
        return sdss + vista_euclid
    elif filter_selection == 'all':
        return all_filters
    else:
        raise ValueError(f'Filter selection {filter_selection} not recognized')


def add_maggies_cols(input_df: Union[pd.DataFrame, pd.Series], fset: FilterSet
                     ) -> Union[pd.DataFrame, pd.Series]:
    """Add maggies column to calalogue of real galaxies.

    Args:
        input_df: either full dataframe galaxy catalogue, or single row (pd.Series)
        fset: the FilterSet used to make the observations.
    """
    df = input_df.copy()  # we don't modify the df inplace
    # Assuming filled values for all 'reliable' filters does not work; instead,
    # we only use only Euclid
    filters = get_filters(fset)
    logging.info(f'Adding maggies cols for {fset}')
    for f in filters:
        mc = df[f.mag_col]
        assert mc is not None
        if isinstance(df, pd.DataFrame):
            df[f.maggie_col] = mc.apply(mags_to_maggies)
        else:
            df[f.maggie_col] = mags_to_maggies(mc)
        mec = df[[f.mag_error_col, f.maggie_col]]
        assert mec is not None
        if isinstance(df, pd.DataFrame):
            df[f.maggie_error_col] = mec.apply(
                    lambda x: calculate_maggie_uncertainty(*x), axis=1)
        else:
            df[f.maggie_error_col] = calculate_maggie_uncertainty(mec[0], mec[1])
    logging.info('Completed adding maggies cols.')
    return df


def mags_to_maggies(mags):
    # mags should be apparent AB magnitudes
    # The units of the fluxes need to be maggies (Jy/3631)
    return 10**(-0.4*mags)


def calculate_maggie_uncertainty(mag_error, maggie):
    # http://slittlefair.staff.shef.ac.uk/teaching/phy217/lectures/stats/L18/index.html#magnitudes
    return maggie * mag_error / 1.09


def load_catalogue(catalogue_loc: str, filters: FilterSet,
                   compute_maggies_cols: bool = False) -> pd.DataFrame:
    """Load a catalogue of photometric observations.

    Regrettably, this function is a little brittle in the sense that we expect
    certain catalogues with certain columns. If this function fails on a new
    catalogue, then alter the *_required_cols (perhaps provide them as an
    argument to this function).

    Args:
        catalogue_loc: file path to the catalogue on disk
        filters: filters used
        compute_maggies_cols: whether to compute maggie_* columns from mag_*
            columns.

    Returns:
        pd.DataFrame: the loaded catalogue.
    """
    logging.info(f'Using {catalogue_loc} as catalogue')

    fs = get_filters(filter_selection=filters)

    maggie_required_cols = [f.maggie_col for f in fs] + \
                           [f.maggie_error_col for f in fs] + \
                           ['redshift']
    mag_required_cols = [f.mag_col for f in fs] + \
                        [f.mag_error_col for f in fs] + \
                        ['redshift']

    if catalogue_loc.endswith('.fits'):
        with fits.open(catalogue_loc) as f:
            df = pd.DataFrame(f[1].data)
            if compute_maggies_cols:
                df = add_maggies_cols(df, filters)
                return df[maggie_required_cols]
            else:
                return df[mag_required_cols]
    else:
        maggie_required_cols += columns.cpz_cols['metadata'] + \
                                columns.cpz_cols['random_forest']
        mag_required_cols += columns.cpz_cols['metadata'] + \
                             columns.cpz_cols['random_forest']

        if catalogue_loc.endswith('.parquet'):
            df = pd.read_parquet(catalogue_loc)
            assert isinstance(df, pd.DataFrame)
            if compute_maggies_cols:
                df = add_maggies_cols(df, filters)
        else:
            if compute_maggies_cols:
                df = pd.read_csv(catalogue_loc, usecols=maggie_required_cols)
            else:
                df = pd.read_csv(catalogue_loc, usecols=mag_required_cols)

        assert isinstance(df, pd.DataFrame)
        if compute_maggies_cols:
            df = df.dropna(subset=maggie_required_cols)
        else:
            df = df.dropna(subset=mag_required_cols)

        assert df is not None
        df_with_spectral_z = df[
            ~pd.isnull(df['redshift'])
        ].query('redshift > 1e-2').query('redshift < 4').reset_index()
        return df_with_spectral_z


def load_galaxy(catalogue_loc: str, filters: FilterSet = Filters.Euclid,
                index: Optional[int] = None, forest_class: Optional[str] = None,
                spectro_class: Optional[str] = None,
                ) -> tuple[pd.Series, int]:
    """Load a galaxy from a catalogue of real-world observations.

    Args:
        catalogue_loc: the filepath to the .fits, .csv or .parquer file
        filters: the filters used in the survey
        index: the optional index of the galaxy to return. If omitted, index is
            random
        forest_class: optional
        spectro_class: optional

    Returns:
        pd.Series: the galaxy's photometry
    """
    df = load_catalogue(catalogue_loc, filters=filters, compute_maggies_cols=False)

    if forest_class is not None:
        logging.warning(f'Selecting forest-identified {forest_class} galaxies')
        df = df.sort_values(f'Pr[{forest_class}]_case_III', ascending=False)
        assert df is not None
        df = df.reset_index(drop=True)
    if spectro_class is not None:
        logging.warning(f'Selecting spectro-identified {spectro_class} galaxies')
        df = df.query(f'hclass == {spectro_class}')
        assert df is not None
        df = df.reset_index()
    assert df is not None
    df.reset_index(drop=True)
    if index is None:
        index = random.randint(0, len(df))

    df_series = add_maggies_cols(df.iloc[index], filters)
    assert isinstance(df_series, pd.Series)
    return df_series, index


def filter_has_valid_data(f: Filter, galaxy: pd.Series) -> bool:
    """Ensures that galaxy data series has maggie cols"""
    filter_value = galaxy[f.maggie_col]
    assert isinstance(filter_value, np.floating)
    valid_value = not pd.isnull(filter_value) \
                  and filter_value > -98 \
                  and filter_value < 98
    filter_error = galaxy[f.maggie_error_col]
    assert isinstance(filter_error , np.floating)
    valid_error = not pd.isnull(filter_error) \
                  and filter_error > 0  # <0 if -99 (unknown) or -1 (only upper bound)
    return bool(valid_value and valid_error)


def load_maggies_to_array(galaxy: pd.Series, filters: list[Filter]
                         ) -> tuple[np.ndarray, np.ndarray]:
    maggies = np.array([galaxy[f.maggie_col] for f in filters])
    maggies_unc = np.array([galaxy[f.maggie_error_col] for f in filters])
    return maggies, maggies_unc


def load_galaxy_for_prospector(
        galaxy: pd.Series, filter_selection: FilterSet
    ) -> tuple[list[observate.Filter], np.ndarray, np.ndarray]:
    all_filters = get_filters(filter_selection)
    valid_filters = [f for f in all_filters if filter_has_valid_data(f, galaxy)]

    if filter_selection == Filters.Reliable and len(valid_filters) != 12:
        raise ValueError(
            f'Some reliable bands are missing - only got {valid_filters}')
    elif filter_selection == Filters.Euclid and len(valid_filters) != 8:
        raise ValueError(
            f'Need 8 valid Euclid bands - only got {valid_filters}')
    elif filter_selection == Filters.DES and len(valid_filters) != 7:
        raise ValueError(
            f'Need 7 valid DES bands - only got {valid_filters}')
    maggies, maggies_unc = load_maggies_to_array(galaxy, valid_filters)
    filters = observate.load_filters([f.bandpass_file for f in valid_filters])
    return filters, maggies, maggies_unc


def load_dummy_galaxy(filter_selection: FilterSet
        ) -> tuple[list[observate.Filter], np.ndarray, np.ndarray]:
    """Loads a dummy galaxy for prospector.

    Args:
        filter_selection: The named filter selection.

    Returns:
        list[observate.Filter]: A list of the loaded sedpy filters.
    """
    filters = get_filters(filter_selection)
    loaded_filters = observate.load_filters([f.bandpass_file for f in filters])
    maggies = np.ones(len(loaded_filters))
    maggies_unc = np.ones(len(loaded_filters))
    return loaded_filters, maggies, maggies_unc
