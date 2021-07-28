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
"""Abstract base class for parameter inference from photometry."""

import abc


class AGNInference(metaclass=abc.ABCMeta):
    """AGNInference is a class implemeting standard methods for inferring
    galaxy parameters from photometric observations.
    """

    def __init__(self):
        # TODO complete this
        pass

    @abc.abstractmethod
    def estimate_params(self):
        # TODO design this function
        raise NotImplementedError
