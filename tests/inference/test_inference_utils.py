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
"""Test inference utilities."""

import torch as t

from agnfinder.types import Tensor
from agnfinder.inference.utils import GalaxyDataset

fpath = './data/testdata/photometry_simulation_1000n_z_0p0000_to_4p0000.hdf5'

def test_GalaxyDaaset():
    dset = GalaxyDataset(file=fpath)
    assert len(dset) == 1000

    xs, ys = dset[500]

    assert isinstance(xs, Tensor)
    assert xs.shape == t.Size([1, 9])
    assert isinstance(ys, Tensor)
    assert ys.shape == t.Size([1, 8])

    xs, ys = dset[[2**i for i in range(5)]]
    assert xs.shape == t.Size([5, 9])
    assert ys.shape == t.Size([5, 8])

    xs, ys = dset[t.randint(0, len(dset), (200,))]
    assert xs.shape == t.Size([200, 9])
    assert ys.shape == t.Size([200, 8])

    t1 = t.from_numpy
    t2 = lambda x: x.to(dtype=t.float64)
    t3 = lambda x: x*0
    t4 = lambda x: x+10

    dset = GalaxyDataset(file=fpath, transforms=[t1, t2, t3, t4])
    xs, ys = dset[0]
    assert isinstance(xs, Tensor)
    assert isinstance(ys, Tensor)
    assert t.equal(xs, t.ones((1, 9), dtype=t.float64)*10)
    assert t.equal(ys, t.ones((1, 8), dtype=t.float64)*10)
