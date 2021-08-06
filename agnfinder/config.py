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
""" Project configuration file """

import os
import math
import torch as t
import logging
from logging.config import dictConfig

from agnfinder.types import ConfigClass, paramspace_t, \
        MaybeFloat, Free, Just, \
        Optional, OptionalValue, Nothing


# ============================= Free Parameters ===============================


class FreeParameters(ConfigClass):
    # Keys prefixed by 'log_*' will be exponentiated later.
    redshift: tuple[float, float] = (0., 4.)
    # Mass of the galaxy
    log_mass: tuple[float, float] = (8, 12)
    # Brightness of the galaxy, from 10**-7 to 15
    log_agn_mass: tuple[float, float] = (-7, math.log10(15))
    # Scale of the agn torus
    log_agn_torus_mass: tuple[float, float] = (-7, math.log10(15))
    dust2: tuple[float, float] = (0., 2.)
    # bounds on this could be made tighter.
    tage: tuple[float, float] = (0.001, 13.8)
    # log_tau > 2, has little effect
    log_tau: tuple[float, float] = (math.log10(.1), math.log10(30))
    agn_eb_v: tuple[float, float] = (0., 0.5)
    inclination: tuple[float, float] = (0., 90.)


# =========================== Sampling Parameters =============================

# These defaults can be overridden by command line arguments when invoking
# agnfinder/simulation/simulation.py (run with --help flag to see options)
class SamplingParams(ConfigClass):
    n_samples: int = 1000
    redshift_min: float = 0.
    redshift_max: float = 4.
    save_dir: str = './data'
    noise: bool = False
    filters: str = 'euclid'


# ============================= CPz Parameters ================================


class CPzParams(ConfigClass):
    """CPz model parameters

    Boolean attributes can be be turned on or off as you please.

    Attributes of type MaybeFloat must either be
        - Just(value), where `value` is a floating point number.
        - Free

    Attributes of type Optional must either be
        - OptionalValue(<MaybeFloat value>)
        - Nothing

    (These monadic data types are defined in types.py.)
    """

    # boolean values {True | False}
    dust: bool = True
    model_agn: bool = True
    igm_absorbtion: bool = True

    # Non-optional values {Free | Just(<float>)}
    agn_mass: MaybeFloat = Free
    redshift: MaybeFloat = Free  # this could be Optional
    inclination: MaybeFloat = Free
    fixed_metallicity: MaybeFloat = Just(0.)  # solar metallicity

    # Optional values
    # {Nothing | OptionalValue(Free) | OptionalValue(Just(<float>))}
    agn_eb_v: Optional = OptionalValue(Free)
    agn_torus_mass: Optional = OptionalValue(Free)


# ============================== SPS Parameters ==============================


class SPSParams(ConfigClass):
    """Parameters for FSPS SSP"""
    zcontinuous: int = 1
    vectoair_flag: bool = False
    compute_vega_mags: bool = False

    # Outdated parameter: allows to emulate FSPS simple stellar population
    # using e.g. a GP (or some other function approximator).
    emulate_ssp: bool = False


# ======================== Quasar Template Parameters =========================


class QuasarTemplateParams(ConfigClass):
    results_dir: str = 'results'

    # quasar parameters
    recreate_quasar_template: bool = True
    quasar_data_loc: str = './data/quasar_template_shang.txt'
    interpolated_quasar_loc: str = './data/quasar_template_interpolated.dill'

    # dusty torus model paramteres
    recreate_torus_template: bool = True
    torus_data_loc: str = './data/clumpy_models_201410_tvavg.hdf5'
    interpolated_torus_loc: str = './data/normalised_torus_model.dill'

    # -> assumptions (fixed parameters) for torus model
    torus_n0: int = 5
    torus_opening_angle:int = 30  # in degrees
    torus_q: int = 2
    torus_y: int = 30
    torus_tv: int = 60

    def results_path(self, file: str) -> str:
        return os.path.join(self.results_dir, file)


# ====================== Extinction Template Parameters =======================


class ExtinctionTemplateParams(ConfigClass):
    results_dir: str = 'results'
    recreate_extinction_template: bool = True
    interpolated_smc_extinction_loc: str \
                       = './data/interpolated_smc_extinction.dill'
    smc_data_loc: str = './data/smc_extinction_prevot_1984.dat'

    def results_path(self, file: str) -> str:
        return os.path.join(self.results_dir, file)


# =========================== Logging Parameters ==============================

# logging levels:
# CRITICAL (50) > ERROR (40) > WARNING (30) > INFO (20) > DEBUG (10) > NOTSET

logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    # see: https://docs.python.org/3/library/logging.html#logrecord-attributes
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(module)s: %(message)s'
        },
        'debug': {
            'format': '[Debugging %(relativeCreated)dms: %(levelname)s] %(filename)s:%(funcName)s:%(lineno)d: %(message)s'
        },
    },
    'handlers': {
        # only log errors out to the console
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'WARNING',
            'stream': 'ext://sys.stderr'
        },
        'console_debug': {
            'class': 'logging.StreamHandler',
            'formatter': 'debug',
            'level': 'DEBUG',
            # print everything to stdout
            'stream': 'ext://sys.stdout'
        },
        # Output more information to a file for post-hoc analysis
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'standard',
            'level': 'INFO',
            'filename': './logs.txt',
            'mode': 'a',
            'encoding': 'utf-8',
            'maxBytes': 500000,
            'backupCount': 4
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file', 'console_debug'],
            'level': 'NOTSET',
            'propagate': True
        }
    }
}


# Utility -------------------------------------------------------------


def configure_logging() -> None:
    """Performs one-time configuration of the root logger for the program.
    """
    dictConfig(logging_config)
    logging.info(
        f'\n\n\n\n\n{35*"~"} New Run {35*"~"}\n\n')


class FreeParams(FreeParameters):
    def __init__(self):

        raw_members = [
            a for a in dir(self)
            if not callable(getattr(self, a))
            and not a.startswith("__")]

        self.raw_params: paramspace_t = {}
        self.params: t.Tensor = t.empty((0, 2), dtype=t.float64)
        self.log: t.Tensor = t.empty((0, 1), dtype=t.bool)

        for m in raw_members:
            self.raw_params[m] = getattr(self, m)
            self.log = t.cat(
                (self.log, t.tensor([[1 if m.startswith('log') else 0]],
                                    dtype=t.bool)))
            self.params = t.cat(
                (self.params, t.tensor([getattr(self, m)], dtype=t.float64)))

        # Remove unnecessary singleton dimension in mask
        self.log = self.log.squeeze(-1)

    def __len__(self) -> int:
        return len(self.raw_params)
