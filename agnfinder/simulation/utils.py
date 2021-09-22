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
"""Utilities for the main simulation class."""

import os
import h5py
import glob
import pyDOE2
import shutil
import logging
import torch as t
import numpy as np

from typing import Any

import argparse
from agnfinder.types import Tensor
from agnfinder.config import FreeParams, SamplingParams


def argparse_to_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Extracts parsed command line arguments as keyword arguments to pass to
    Simulator constructor.

    Args:
        args: parsed command line arguments

    Returns:
        dict[str, Any]: kwargs for Simulator constructor.
    """
    return {
        'n_samples': args.n_samples,
        'save_dir': args.save_dir,
        'filters': args.filters,
        'emulate_ssp': args.emulate_ssp,
        'noise': args.noise
    }


def get_unit_latin_hypercube(dims: int, n_samples: int) -> Tensor:
    """Generate latin hypercube.

    Only reason to use pyDOE2 to do this is for the correlation minimisation.

    Args:
        dims: The number of dimensions of the hypercube.
        n_samples: Number of rows in the resulting matrix.

    Returns:
        np.ndarray: A n_samples x dims numpy array, with one row for each
            sample.
    """
    np_cube = pyDOE2.lhs(n=dims, samples=n_samples, criterion='correlation')
    return t.from_numpy(np_cube)


def shift_redshift_theta(norm_redshift: Tensor,
                         fixed_theta_range: tuple[float, float],
                         target_theta_range: tuple[float, float]) -> Tensor:
    """Transforms redshift range so that when this hypercube is used with other
    hypercubes (with other redshift ranges), the normalised redshift will still
    cover the [0, 1] range when combined.

    When denormalised, the transformed redshift will cover the target redshift
    range for this cube.

    Args:
        norm_redshift: Normalised redshift samples (in unit interval)
        fixed_theta_range: Range for this hypercube (config.py:free_params)
        target_theta_range: Target range (config.py:SamplingParams or CLI args)

    Returns:
        np.ndarray: The shifted redshift.

    Example:
        norm_redshift necessarily lies in [0, 1]. If target_theta_range =
        [1.2, 1.8], then we return an array covering [0.4, 0.6], so that when
        'denormalised' (i.e. scaled and offset by fixed_theta_range) it will
        lie in correct range.
    """
    rshift = norm_redshift * \
          (target_theta_range[1] - target_theta_range[0]) \
        / (fixed_theta_range[1] - fixed_theta_range[0])

    # Adjust the start in norm space
    return rshift + target_theta_range[0] / fixed_theta_range[1]


def denormalise_theta(norm_theta: Tensor, limits: FreeParams) -> Tensor:
    """Convert hypercube to the true parameter space.

    Args:
        norm_theta: Normalised hypercube of parameter values
        limits: Free parameter definition (including limits)

    Returns:
        Tensor: Scaled parameter values
    """
    assert norm_theta.shape[1] == len(limits)

    # rescale parameters
    theta = limits.params[:,0] + \
            (limits.params[:,1]-limits.params[:,0]) * norm_theta

    # exponentiate log parameters
    return t.where(limits.log, 10**t.clip(theta, -10, 20), theta)


def normalise_theta(params: Tensor, limits: FreeParams) -> Tensor:
    """Normalise galaxy parameters to lie in range [0, 1]

    Useful for turning Prospector samples into normalised samples.

    Args:
        params: The hypercube of galaxy parameters.
        limits: Free paameter definition (including limits)

    Returns:
        Tensor: The normalised SPS parameters
    """
    assert params.shape[1] == len(limits)

    # Take logarithm for logarithmic parameters.
    params = t.where(limits.log, t.log10(params), params)

    # 'Squeeze' parameters by their ranges and offset to lie in unit interval
    return (params - limits.params[:,0]) / \
            (limits.params[:,1]-limits.params[:,0])


def ensure_partials_dir(save_dir: str) -> None:
    """A utility function to ensure that that partials directory (to store
    intermediate results) is created and empty.

    Args:
        save_dir: base location where the final model will be stored
    """
    path = os.path.join(save_dir, "partials")
    if os.path.exists(path):
        logging.warning(f'Removing existing partials directory at {path}')
        shutil.rmtree(path)
    os.mkdir(path)


def _must_get_grp(f: h5py.File, key: str) -> h5py.Group:
    g = f.get(key)
    assert g is not None and isinstance(g, h5py.Group)
    return g


def _must_get_dset(g: h5py.Group, key: str) -> h5py.Dataset:
    d = g.get(key)
    assert d is not None and isinstance(d, h5py.Dataset)
    return d


def join_partial_samples(sp: SamplingParams) -> None:
    """Joins all the intermediate results (partials/*.hdf5) into a single file,
    and removes the partial results.

    Args:
        sp: the simulation paramteres

    Raises:
        RuntimeError: if no result files are found in partials directory.

    Note:
        It is possible to include links to external files in hdf5. If this
        approach of copying values to a single file is too slow / memory
        intensive, we could just use links...
    """
    fp = FreeParams()

    zmin_str = f'{sp.redshift_min:.4f}'.replace('.', 'p')
    zmax_str = f'{sp.redshift_max:.4f}'.replace('.', 'p')
    save_name = 'photometry_simulation_{}n_z_{}_to_{}.hdf5'.format(
        sp.n_samples, zmin_str, zmax_str
    )
    save_path = os.path.join(sp.save_dir, save_name)
    partial_dir = os.path.join(sp.save_dir, "partials")

    pfiles = glob.glob(partial_dir+"/*.hdf5")
    if len(pfiles) == 0:
        raise RuntimeError(f'No partial result files found in {partial_dir}')
    if len(pfiles) == 1:
        # just move the file in the singleton list
        os.rename(pfiles[0], save_path)
        logging.info(f'Saved samples to {save_path}')
        shutil.rmtree(partial_dir)
        return

    def _map_concat(arrs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        assert all(map(lambda x: isinstance(x, np.ndarray), arrs))
        ret = np.concatenate(arrs, 0)
        assert isinstance(ret, np.ndarray)
        return ret

    DS: list[np.ndarray] = []
    ds_keys = ['theta', 'normalised_theta', 'simulated_y', 'wavelengths']
    for i, p in enumerate(pfiles):
        s = _must_get_grp(h5py.File(p, 'r'), 'samples')
        if i == 0:
            DS = list(map(lambda x: np.array(_must_get_dset(s, x)), ds_keys))
        else:
            D = list(map(lambda x: np.array(_must_get_dset(s, x)), ds_keys))
            DS = list(map(_map_concat, zip(DS, D)))

    [theta, theta_norm, phot, wlengths] = DS

    with h5py.File(save_path, 'w') as f:
        grp = f.create_group('samples')
        ds_x = grp.create_dataset('theta', data=theta)
        ds_x.attrs['columns'] = fp.raw_members
        ds_x.attrs['description'] = 'Parameters used by simulator'

        ds_x_norm = grp.create_dataset('normalised_theta', data=theta_norm)
        ds_x_norm.attrs['columns'] = fp.raw_members
        ds_x_norm.attrs['description'] = \
            'Normalised parameters used by simulator'

        ds_y = grp.create_dataset('simulated_y', data=phot)
        ds_y.attrs['description'] = 'Response of simulator'

        ds_wavelengths = grp.create_dataset('wavelengths', data=wlengths)
        ds_wavelengths.attrs['description']= \
            'Observer wavelengths to visualise simulator photometry'
    logging.info(f'Saved samples to {save_path}')

    logging.info(f'Removing partial results in {partial_dir}')
    shutil.rmtree(partial_dir)
