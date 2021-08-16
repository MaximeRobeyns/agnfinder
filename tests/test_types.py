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
"""Tests for some of the custom data types."""

import pytest
import torch.nn as nn

from agnfinder.types import Filters, FilterSet, arch_t

def test_Filters():

    f: FilterSet = Filters.Euclid
    assert f.dim == 8
    assert f.value == 'euclid'
    assert f == 'euclid'

    f = Filters.Reliable
    assert f.dim == 12
    assert f.value == 'reliable'
    assert f == 'reliable'

    f = Filters.All
    assert f.dim == 12
    assert f.value == 'all'
    assert f == 'all'

def test_arch_t():

    with pytest.raises(ValueError):
        # test len(layer_sizes) /= len(activations)
        _ = arch_t([1,2,3], [4], [nn.ReLU()])

    # test type(activations) must be list[nn.Module]
    with pytest.raises(ValueError):
        _ = arch_t([1,2,3], [4], [None])
    with pytest.raises(ValueError):
        _ = arch_t([1,2,3], [4], [42])

    # test type(activations) must be nn.Module
    with pytest.raises(ValueError):
        _ = arch_t([1,2,3], [4], None)
    with pytest.raises(ValueError):
        _ = arch_t([1,2,3], [4], 42)

    # Verify that this doesn't raise a ValueError
    a = arch_t([1,2,3], [4], nn.ReLU())
    assert len(a) == 4
    assert len(a.activations) == 3

    aa = arch_t([1,2], [3, 4], [nn.ReLU(), nn.Softmax(1)])
    assert len(aa) == 3
    assert len(aa.activations) == 2
    assert len(aa.layer_sizes) == 2
    assert len(aa.head_sizes) == 2
    assert all([a is None for a in aa.head_activations])
    assert aa.batch_norm

    aaa = arch_t([2**i for i in range(10, 5, -1)],
                 [10, 2], nn.ReLU(), [None, nn.Softmax()], False)
    assert len(aaa) == 6
    assert not aaa.batch_norm
