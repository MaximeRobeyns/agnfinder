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
"""Tests for the simulation class"""

import pytest
import argparse
import torch as t

from agnfinder.simulation import Simulator
from agnfinder.config import FreeParams

def _create_test_sim() -> Simulator:
    kwargs = {
        'n_samples': 1000,
        'rshift_min': 0.,
        'rshift_max': 4.,
        'save_dir': './data/testdata',
        'emulate_ssp': False,
        'noise': False,
        'filters': 'euclid'
    }
    testargs = argparse.Namespace(**kwargs)
    fp = FreeParams()

    return Simulator(testargs, fp)

def test_simulation():

    sim = _create_test_sim()

    assert sim.n_samples == 1000
    assert sim.save_name == \
        'photometry_simulation_1000n_z_0p0000_to_4p0000.hdf5'
    assert sim.rshift_range == (0., 4.)

    assert not sim.hcube_sampled
    assert not sim.has_forward_model
    assert not sim.has_run

def test_create_hypercube():

    sim = _create_test_sim()
    fp = FreeParams()
    n_free = len(fp)

    sim.sample_theta()

    assert sim.hcube_sampled
    assert sim.theta.shape == t.Size((sim.n_samples, n_free))

    # We have already tested that the denormalised theta are in the correct
    # range in test_utils.py

@pytest.mark.slow
def test_run_and_save_model():
    sim = _create_test_sim()
    assert not sim.has_forward_model
    sim.create_forward_model()
    assert sim.has_forward_model
    sim.run()
    sim.save_samples()
