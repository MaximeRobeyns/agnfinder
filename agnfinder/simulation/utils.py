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

import pyDOE2

import torch as t

from agnfinder.types import Tensor
from agnfinder.config import FreeParams


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
