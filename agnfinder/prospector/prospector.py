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
"""Main Prospector problem generation class."""

# TODO configure logging
import logging

from agnfinder.config import CPzParams
from agnfinder.prospector import cpz_builders


class Prospector(object):

    def __init__(self, filter_selection: str, emulate_ssp: bool):

        self.obs = cpz_builders.build_cpz_obs(filter_selection=filter_selection)
        logging.info(self.obs)

        cpz_params = CPzParams()

        self.model = cpz_builders.build_model(cpz_params)
        logging.info(self.model)

        self.sps = cpz_builders.build_sps(cpz_params, emulate_ssp, zcontinuous=1)
        logging.info(self.sps)

