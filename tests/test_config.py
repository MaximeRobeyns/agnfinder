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
"""Tests for the project configuration file"""

import agnfinder.config as config


def test_FreeParams():
    assert len(config.free_params) == 9

    fp = config.FreeParams(config.free_params)
    for p, k in enumerate(fp.raw_params):
        if k.startswith('log'):
            assert fp.log[p]
        else:
            assert not fp.log[p]
        assert fp.params[p][0] == fp.raw_params[k][0]
        assert fp.params[p][1] == fp.raw_params[k][1]
        assert hasattr(fp, k)

    assert fp.log.shape == (9,)
    assert fp.params.shape == (9,2)


