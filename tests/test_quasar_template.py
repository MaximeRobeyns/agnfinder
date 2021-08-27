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
"""Tests for the quasar templates."""

import os
import time
import pytest
import agnfinder.config as cfg
from agnfinder.quasar_templates import QuasarTemplate, TorusModel


# This test assumes that the user has carried out the full installation
# procedure, and that they have the 1.2Gb clumpy model available.

def test_create_QuasarTemplate():
    params = cfg.QuasarTemplateParams()

    qt = QuasarTemplate(
        template_loc=params.interpolated_quasar_loc,
        data_loc=params.quasar_data_loc,
        recreate_template=True)

    assert not qt._load_error
    assert qt._interpolated_template is not None

    with pytest.raises(ValueError) as re:
        qt = QuasarTemplate(
            template_loc=params.interpolated_quasar_loc,
            data_loc="./data/asdf",
            recreate_template=True)
        assert re == "Data location ./data/asdf does not exist"


def test_load_QuasarTemplate():
    params = cfg.QuasarTemplateParams()
    qt = QuasarTemplate(template_loc=params.interpolated_quasar_loc)
    assert not qt._load_error
    assert qt._interpolated_template is not None


def test_create_TorusModel():
    now = time.time()
    params = cfg.QuasarTemplateParams()
    tm = TorusModel(
        params=params,
        template_loc=params.interpolated_torus_loc,
        data_loc=params.torus_data_loc,
        recreate_template=True)
    assert not tm._load_error
    assert tm._interpolated_template is not None
    assert os.path.exists(params.interpolated_torus_loc)
    assert os.path.getmtime(params.interpolated_torus_loc) - now > 0


def test_load_TorusModel():
    params = cfg.QuasarTemplateParams()
    assert os.path.exists(params.interpolated_torus_loc)
    tm = TorusModel(params, template_loc=params.interpolated_torus_loc)
    assert tm._interpolated_template is not None
