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
from typing import Any, Optional

from agnfinder.types import FilterSet
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
    def catalogue_loc(self) -> str:
        """The path to the catalogue of observations. Either .csv, .parquet or
        .fits"""
        pass


# Dynesty Description ---------------------------------------------------------


class DynestyParams(MCMCParams):
    """Configuration specifc to Dynesty (nested sampling)

    Note: this UltraNest is ostensibly a bit better (see UltraNest docs).
    """

    @property
    def method(self) -> str:
        return 'rwalk'
    @property
    def bound(self) -> str:
        return 'multi'

    @property
    def bootstrap(self) -> int:
        return 0

    @property
    def sample(self) -> str:
        return 'unif'

    @property
    def nlive_init(self) -> int:
        return 400

    @property
    def nlive_batch(self) -> int:
        return 200

    @property
    def weight_kwargs(self) -> dict[str, Any]:
        return {"pfrac": 1.0}

    @property
    def stop_kwargs(self) -> dict[str, Any]:
        return {"post_thresh": 0.1}

    @property
    def dlogz_init(self) -> float:
        return 0.05

    @property
    def posterior_thresh(self) -> float:
        return 0.05

    @property
    def maxcall(self) -> int:
        return 200

    @property
    def optimize(self) -> bool:
        return False

    @property
    def nmin(self) -> int:
        return 10

    @property
    def min_method(self) -> str:
        # not optimising....
        return ''  # 'lm' | 'powell'


# EMCEE Description -----------------------------------------------------------


class EMCEEParams(MCMCParams):
    """Configuration specific to EMCEE (with some default values)

    See https://arxiv.org/abs/1202.3665 (section 4) for for tips on setting
    these parameters.
    """

    @property
    def nwalkers(self) -> int:
        """The number of 'walkers'. The more the better."""
        return 128

    @property
    def nburn(self) -> list[int]:
        """Burn-in (per walker)"""
        return [512]

    @property
    def niter(self) -> int:
        return 10000

    @property
    def interval(self) -> float:
        return .25

    @property
    def initial_disp(self) -> float:
        return .1

    @property
    def optimize(self) -> bool:
        return True

    @property
    def nmin(self) -> int:
        return 10

    @property
    def min_method(self) -> str:
        """lm | powell"""
        return 'powell'


# UltraNest Description -------------------------------------------------------


class UltraNestParams(MCMCParams):

    # sampler specific parameters ---------------------------------------------

    @property
    def ndraw_min(self) -> int:
        """minimum number of points to simultaneously propose. Increase this if
        the likelihood makes vectorization very cheap"""
        return 128

    @property
    def ndraw_max(self) -> int:
        """maximum number of points to simultaneously propose. Increase this if
        the likelihood makes vectorization very cheap. Watch out for memory
        issues with very high values."""
        return 65536

    # run specific parameters -------------------------------------------------

    @property
    def update_interval_volume_fraction(self) -> float:
        return 0.8

    @property
    def show_status(self) -> bool:
        """Show integration progress as a status line."""
        return True

    @property
    def viz_callback(self) -> Any:
        return 'auto'
        # return False # 'auto'

    @property
    def dlogz(self) -> float:
        """Target evidence uncertainty. This is the std between bootstrapped
        logz integrators"""
        return 0.5

    @property
    def dKL(self) -> float:
        """Target posterior uncertainty. This is the KL divergence in nats
        between bootstrapped integrators"""
        return 0.5

    @property
    def frac_remain(self) -> float:
        """Integrate until this fraction of the integral is left in the
        remainder.

        Lower numbers (1e-2...1e-5) ensures that peaks are discovered, while
        larger numbers (0.5) are better for simple posteriors.
        """
        return 1e-4

    @property
    def Lepsilon(self) -> float:
        """Terminate when live point likelihoods are all the same within
        Lepsilon tolerance.

        Higher values are better when the likelihood function is inaccurate; to
        avoid unnecessary search.
        """
        return 0.001

    @property
    def min_ess(self) -> int:
        """Target number of effective posterior samples"""
        return 2000

    @property
    def max_iters(self) -> Optional[int]:
        """Maximum number of integration iterations"""
        return None

    @property
    def max_ncalls(self) -> Optional[int]:
        """Stops after this many likelihood evaluations"""
        return None

    @property
    def max_num_improvement_loops(self) -> int:
        """Limits the number of 'improvement loops' for assessing where more
        samples are needed. -1 = no limit"""
        return -1

    @property
    def min_num_live_points(self) -> int:
        """Minimum number of live points throughout the run. At the end of the
        run, you will get at least this many points drawn from the
        posterior."""
        return 400

    @property
    def cluster_num_live_points(self) -> int:
        """require at least this many live points per detected cluster"""
        return 40
