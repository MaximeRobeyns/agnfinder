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
"""MCMC Parameter definition.

The reason that this is in a separate file and not in the mcmc.py file is to
avoid a circular dependency.
"""

import torch as t

from abc import abstractmethod
from typing import Type

from agnfinder.types import FilterSet, MCMCMethod
from agnfinder.inference.inference import ModelParams


# MCMC Description ------------------------------------------------------------


class MCMCParams(ModelParams):
    """Configuration class for MCMC
    """

    epochs: int = -1
    batch_size: int = -1
    dtype: t.dtype = t.float64
    device: t.device = t.device('cpu')

    @property
    @abstractmethod
    def cond_dim(self) -> int:
        """Length of 1D conditioning information vector"""
        pass

    @property
    @abstractmethod
    def data_dim(self) -> int:
        """Length of the perhaps (flattened) 1D data vector, y"""
        pass

    @property
    @abstractmethod
    def filters(self) -> FilterSet:
        """The filters used in the survey; used for initialising the forward
        model."""
        pass

    @property
    def emulate_ssp(self) -> bool:
        return False

    @property
    @abstractmethod
    def inference_procedure(self) -> Type[MCMCMethod]:
        """The inference method to use; either EMCEE or Dynesty."""
        pass

    @property
    @abstractmethod
    def catalogue_loc(self) -> str:
        """The path to the catalogue of observations. Either .csv, .parquet or
        .fits"""
        pass
