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

import agnfinder.config as cfg
from agnfinder.quasar_templates import QuasarTemplate, TorusModel


# Unfortunately don't test the _create_template functions here since this
# requires the user to have access to the original data, which they may not
# have.

def test_QuasarTemplate():

    params = cfg.QuasarTemplateParams()

    qt = QuasarTemplate(template_loc=params.interpolated_torus_loc)



