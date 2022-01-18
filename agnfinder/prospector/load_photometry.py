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

import os
import sys
import h5py
import random
import logging
import torch as t
import numpy as np
import pandas as pd

from sedpy import observate
from typing import Optional, Union, Any, Callable
from astropy.io import fits
from torch.utils.data import Dataset, DataLoader, random_split

from . import columns

from agnfinder.types import FilterSet, Filters, Tensor, tensor_like


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


def get_simulated_galaxy(path: str, filters: FilterSet, index: int = -1,
        add_unc: bool = True, compute_maggies_cols: bool = True
        ) -> tuple[Union[pd.DataFrame, pd.Series], tensor_like]:
    """Get a (single) simulated galaxy from a cube.

    Args:
        path: path to the .hdf5 file containing the simulations or a directory
            of hdf5 files containing sub-cubes.
        filters: the filters used to generate the samples
        index: the required index. If this is < 0, a random galaxy is chosen.
        add_unc: Whether to add in some simulated uncertainties (e.g. as
            required for some MCMC methods).
        compute_maggies_cols: whether to compute maggie_* columns from the
            mag_* columns. Set to true by default as required in most mcmc
            methods.

    Returns:
        tuple[pd.Series, np.ndarray]: simulated galaxy photometry and 'true'
            parameters.
    """

    def _dsets_from_file(f: h5py.File) -> tuple[h5py.Dataset, h5py.Dataset]:
        samples = f.get('samples')
        assert isinstance(samples, h5py.Group)
        sim_y = samples.get('simulated_y')
        norm_theta = samples.get('normalised_theta')
        assert isinstance(sim_y, h5py.Dataset)
        assert isinstance(norm_theta, h5py.Dataset)
        return sim_y, norm_theta

    simulated_obs = norm_theta = None

    # In case of random index for directory of hdf5 files:
    if os.path.isdir(path) and index < 0:
        files = [f for f in os.listdir(path) if f.endswith(".hdf5")]
        assert len(files), f'No hdf5 files found in {path}!'
        path = random.choice(files)

    if os.path.isdir(path):
        ii = jj = 0
        for file in filter(lambda x: x.endswith('.hdf5'), os.listdir(path)):
            y, theta = _dsets_from_file(h5py.File(file))
            ii += len(y)
            if index < ii:
                simulated_obs = y[index - jj]
                norm_theta = theta[index - jj]
                ii = -1; break
            else:
                jj = ii
        if ii < 0:
            raise ValueError(f'Provided index ({index}) > samples ({jj})')
    elif os.path.isfile(path):
        y, theta = _dsets_from_file(h5py.File(path, 'r'))
        if index > len(y):
            raise ValueError(f'Provided index ({index}) > samples ({len(y)})')
        if index >= 0 and index < len(y):
            simulated_obs = y[index]
            norm_theta = theta[index]
    else:
        logging.error((f'Provided path ({path}) is neither a directory or '
                       f'path to hdf5 file.'))
        sys.exit()

    # Get the filters
    fs = get_filters(filter_selection=filters)

    phot = pd.Series(simulated_obs, [f.mag_col for f in fs])

    if add_unc:
        # add initial 'uncertainties' to photometry.
        unc = pd.Series(map (lambda x: x*0.00001, phot.values),
                       [f.mag_error_col for f in fs])
        phot = phot.combine_first(unc)

    if compute_maggies_cols:
        # MCMC requires maggies and maggie uncertainties.
        phot = add_maggies_cols(phot, filters)

    assert isinstance(norm_theta, np.ndarray)
    assert isinstance(phot, pd.Series) or isinstance(phot, pd.DataFrame)

    return phot, norm_theta


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


class GalaxyDataset(Dataset):

    def __init__(self, path: str, filters: FilterSet,
                 transforms: list[Callable[[Any], Any]] = [],
                 x_transforms: list[Callable[[Any], Any]] = [],
                 y_transforms: list[Callable[[Any], Any]] = []) -> None:
        """PyTorch Dataset for galaxy observations.

        Args:
            path: either the path of a hdf5 / fits file containing catalogue.
            filters: filters to use with this dataset
            transforms: any transformations to apply to the data now
            x_transforms: photometry specific transforms
            y_transforms: parameter specific transforms
        """

        self.transforms = transforms
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms

        xs, ys = self._get_x_y_from_file(path, filters)

        # eagerly compute transformations (rather than during __getitem__)
        for xtr in self.x_transforms:
            xs = xtr(xs)
        for ytr in self.y_transforms:
            ys = ytr(ys)

        self._x_dim, self._y_dim = xs.shape[-1], ys.shape[-1]
        self.dataset = np.concatenate((xs, ys), -1)
        logging.info('Galaxy dataset loaded')

    def _get_x_y_from_file(self, path: str, filters: FilterSet
                           ) -> tuple[np.ndarray, np.ndarray]:
        assert os.path.exists(path)

        df = load_catalogue(path, filters=filters, compute_maggies_cols=True)
        assert df is not None

        fs = get_filters(filters)
        xs_cols = [f.maggie_col for f in fs]
        ys_col = ['redshift']

        xs, ys = df[xs_cols].to_numpy(), df[ys_col].to_numpy()

        return xs, ys

    def get_xs(self) -> Any:
        """Just return all the xs (photometric observations) in the dataset"""
        xs = self.dataset[:, :self._x_dim]
        for tr in self.transforms:
            xs = tr(xs)
        return xs.squeeze()

    def get_ys(self) -> Any:
        """Return all the y values in the dataset"""
        ys = self.dataset[:, self._x_dim:]
        for tr in self.transforms:
            ys = tr(ys)
        return ys

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: Union[int, list[int], Tensor]) -> tuple[
            tensor_like, tensor_like]:

        if isinstance(idx, Tensor):
            idx = idx.to(dtype=t.int).tolist()

        data = self.dataset[idx]
        if data.ndim == 1:
            data = np.expand_dims(data, 0)
        xs, ys = data[:, :self._x_dim], data[:, self._x_dim:]

        for tr in self.transforms:
            xs, ys = tr(xs).squeeze(), tr(ys).squeeze()

        # if ys.dim() < 2:
        #     ys = ys.unsqueeze(-1)

        return xs, ys


def load_real_data(
        path: str, filters: FilterSet, split_ratio: float=0.8,
        batch_size: int=32, test_batch_size: Optional[int]=None,
        transforms: list[Callable[[Any], Any]] = [t.from_numpy],
        x_transforms: list[Callable[[Any], Any]] = [],
        y_transforms: list[Callable[[Any], Any]] = [],
        split_seed: int = 0
        ) -> tuple[DataLoader, DataLoader]:
    """Load real data as PyTorch DataLoaders.

    Since we only usually have access to the redshift parameter and not any
    other physical parameters, the xs are the photometric observations, and the
    ys are just the redshift values.

    Args:
        path: file path to the .fits / .hdf5 file containing simulated data
        filters: filters used in the catalogue (used to infer required columns)
        split_ratio: train / test split ratio
        batch_size: training batch size (default 32)
        test_batch_size: optional different batch size for testing (defaults to
            `batch_size`)
        transforms: list of transformations to apply to data before returning
        x_transforms: any photometry specific transformations
        y_transforms: any parameter specific transformations
        split_seed: PyTorch Generator Seed for reproducible train/test splits.

    Returns:
        tuple[DataLoader, DataLoader]: train and test DataLoaders, respectively
    """
    tbatch_size = test_batch_size if test_batch_size is not None else batch_size

    cuda_kwargs = {'num_workers': 8, 'pin_memory': True}
    train_kwargs: dict[str, Any] = {
        'batch_size': batch_size, 'shuffle': True} | cuda_kwargs
    test_kwargs: dict[str, Any] = {
        'batch_size': tbatch_size, 'shuffle': True} | cuda_kwargs

    dataset = GalaxyDataset(path, filters, transforms, x_transforms, y_transforms)

    n_train = int(len(dataset) * split_ratio)
    n_test = len(dataset) - n_train

    rng = t.Generator().manual_seed(split_seed)
    train_set, test_set = random_split(dataset, [n_train, n_test], rng)

    train_loader = DataLoader(train_set, **train_kwargs)
    test_loader = DataLoader(test_set, **test_kwargs)

    return train_loader, test_loader


def load_galaxy(catalogue_loc: str, filters: FilterSet = Filters.Euclid,
                index: Optional[int] = None, forest_class: Optional[str] = None,
                spectro_class: Optional[str] = None,
                ) -> tuple[pd.Series, int]:
    """Load a galaxy from a catalogue of real-world observations.

    Args:
        catalogue_loc: the filepath to the .fits, .csv or .parquet file
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


def sample_galaxies(catalogue_loc: str, filters: FilterSet = Filters.Euclid,
                    n_samples: int = 1000, has_redshift: bool = True
                    ) -> Union[pd.Series, pd.DataFrame]:
    """Load a random sample of galaxies from a catalogue of real-world observations.

    Args:
        catalogue_loc: the filepath to the .fits, .csv or .parquet file
            containing the observations.
        filters: the filters used in the survey
        n_samples: the number of galaxies to return. If -1, return all from catalogue.
        has_redshift: only return galaxies with the redshift parameter present
            (default True)

    Returns:
        np.ndarray: an n_samples x data_dim array of observations.
    """
    df = load_catalogue(catalogue_loc, filters=filters, compute_maggies_cols=False)
    assert df is not None
    if has_redshift:
        df = df.loc[df.redshift >= 0]
    df.reset_index(drop=True)
    if n_samples == -1:
        # random permutation of indices
        idxs = np.random.choice(len(df), len(df), replace=False)
    elif n_samples > 0:
        idxs = np.random.choice(len(df), n_samples, replace=False)
    else:
        raise ValueError(f'n_samples ({n_samples})cannot be negative!')
    df_series = add_maggies_cols(df.iloc[idxs], filters)
    assert isinstance(df_series, pd.DataFrame) or isinstance(df_series, pd.Series)
    return df_series


def filter_has_valid_data(f: Filter, galaxy: pd.Series) -> bool:
    """Ensures that galaxy data series has maggie cols"""
    filter_value = galaxy[f.maggie_col]
    assert isinstance(filter_value, np.floating) or isinstance(filter_value, float)
    valid_value = not pd.isnull(filter_value) \
                  and filter_value > -98 \
                  and filter_value < 98
    filter_error = galaxy[f.maggie_error_col]
    assert isinstance(filter_error, np.floating) or isinstance(filter_error, float)
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
