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
"""Tests for the simulation utilities"""

import torch as t

import agnfinder.config as cfg
from agnfinder.simulation import utils


def test_get_unit_latin_hypercube():
    # expect a matrix with 10 rows, and 4 columns
    hcube = utils.get_unit_latin_hypercube(dims=4, n_samples=10)
    assert type(hcube) == t.Tensor

    assert hcube.shape == t.Size((10, 4))
    for i in range(hcube.shape[0]):
        for j in range(hcube.shape[1]):
            assert 0 <= hcube[i][j]
            assert hcube[i][j] <= 1


def test_shift_redshift_theta():
    hcube = utils.get_unit_latin_hypercube(dims=2, n_samples=10)
    zs = hcube[:, 0]
    assert zs.shape == t.Size((10,))

    # comes from config.py:FreeParameters
    fixed_theta_range = (0., 4.)

    # comes from config.py:SamplingParams or CLI arguments
    target_theta_range = (1.2, 1.8)

    zs_new = utils.shift_redshift_theta(
            zs, fixed_theta_range, target_theta_range)

    fr = fixed_theta_range[1] - fixed_theta_range[0]
    tr = target_theta_range[1] - target_theta_range[0]

    lb = target_theta_range[0] / fixed_theta_range[1]
    ub = lb + fr * tr

    assert zs_new.shape == t.Size((10,))
    for z in zs_new:
        assert lb <= z
        assert z <= ub


def test_denormalise_theta():

    fp = cfg.FreeParams()
    dims = len(fp)
    hcube = utils.get_unit_latin_hypercube(dims, 10)

    denorm = utils.denormalise_theta(hcube, fp)

    assert denorm.shape == hcube.shape

    for row in denorm:
        for c in range(len(row)):
            lb, ub = fp.params[c][0].item(), fp.params[c][1].item()
            if fp.log[c]:
                # clip
                lb = 10**max(-10., min(lb, 20.))
                ub = 10**max(-10., min(ub, 20.))
            assert lb <= row[c]
            assert row[c] <= ub


def test_normalise_theta():

    fp = cfg.FreeParams()
    dims = len(fp)
    hcube = utils.get_unit_latin_hypercube(dims, 10)

    denorm = utils.denormalise_theta(hcube, fp)
    norm = utils.normalise_theta(denorm, fp)

    assert hcube.shape == norm.shape

    # Due to clipping in log values, we can't easily verify that hcube == norm
