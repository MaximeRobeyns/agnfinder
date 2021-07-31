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

from sedpy import observate


class Filter(object):
    def __init__(self, bandpass_file: str, mag_col:str, error_col:str):
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


def get_filters(selection: str) -> list[Filter]:
    """Get the list of Filter objects corresponding to the named filter selection.

    Args:
        selection: The filter selection; {reliable, euclid, all}

    Returns:
        list[Filter]: A list all the filters for the selection.

    Raises:
        ValueError: If the selection is not recognised.
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
            bandpass_file=f'{b}_cfhtl',
            mag_col=f'mag_auto_cfhtwide_{b}_dr7',
            error_col=f'magerr_auto_cfhtwide_{b}_dr7')
        for b in ['g', 'i', 'r', 'u', 'z']]

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
        for b in ['w1', 'w2']]

    # These are _not_ in wavelength order.
    all_filters = galex + sdss + cfht + kids + vista + wise

    if selection == 'reliable':
        return sdss + vista + wise
    elif selection == 'euclid':
        return sdss + vista_euclid
    elif selection == 'all':
        return all_filters
    else:
        raise ValueError(f'Filter selection {selection} not recognized')

# TODO add other functions as required
# def add_maggies_cols(input_df: pd.DataFrame) -> pd.DataFrame:
#     df = input_df.copy()
#     filters = get_filters('reliable')
#     for f in tqdm.tqdm(filters):
#         df[f.maggie_col] = df[f.mag_col].apply(mags_to_maggies)
#         df[f.maggie_error_col] = df[[f.mag_error_col, f.maggie_col]].apply(
#                 lambda x: calculate_maggie_uncertainty(*x), axis=1)
#     return df
#
# def mags_to_maggies(mags):
#     # mags should be apparent AB magnitudes
#     # The units of the fluxes need to be maggies (Jy/3631)
#     return 10**(-0.4*mags)
#
#
# def load_galaxy_for_prospector() -> None:
#     # TODO what is a 'galaxy' here?
#     # Is it a DataFrame?
#     raise NotImplementedError


def load_dummy_galaxy(filter_selection: str
        ) -> list[observate.Filter]:
    """Loads a dummy galaxy for prospector.

    Args:
        filter_selection: The named filter selection.

    Returns:
        list[observate.Filter]: A list of the loaded sedpy filters.
    """
    filters = get_filters(selection=filter_selection)
    loaded_filters = observate.load_filters([f.bandpass_file for f in filters])
    return loaded_filters

