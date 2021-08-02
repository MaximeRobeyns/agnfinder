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

# Keys prefixed by 'log_*' will be exponentiated later.
free_params: paramspace_t = {
    'redshift': (0., 4.),
    # Mass of the galaxy
    'log_mass': (8, 12),
    # Brightness of the galaxy
    'log_agn_mass': (-7, math.log10(15)),  # from 10**-7 to 15
    # Scale of the agn torus
    'log_agn_torus_mass': (-7, math.log10(15)),
    'dust2': (0., 2.),
    'tage': (0.001, 13.8),  # bounds on this could be made tighter.
    'log_tau': (math.log10(.1), math.log10(30)),  # >2, has little effect
    'agn_eb_v': (0., 0.5),
    'inclination': (0., 90.)
}


# =========================== Sampling Parameters =============================

# These defaults can be overridden by command line arguments when invoking
# agnfinder/simulation/simulation.py (run with --help flag to see options)
class SamplingParams(ConfigClass):
    n_samples: int = 1000
    redshift_min: float = 0.
    redshift_max: float = 4.
    save_dir: str = './data'
    emulate_ssp: bool = False
    noise: bool = False
    filters: str = 'euclid'


# ============================= CPz Parameters ================================


class CPzParams(ConfigClass):
    """Classification-aided photometric-redshift model parameters

    Attributes with type bool can be turned on or off as you please.

    Attributes of type MaybeFloat must either be

        - Just(value), where `value` is a floating point number.
        - Free

    Attributes of type Optional must either be

        - OptionalValue(<MaybeFloat value>)
        - Nothing

    (These monadic data types are defined in types.py.)
    """
    # TODO (Maxime): document what these parameters actually mean.

    # boolean values {True | False}
    dust: bool = True
    model_agn: bool = True
    igm_absorbtion: bool = True

    # Non-optional values {Free | Just(<float>)}
    agn_mass: MaybeFloat = Free
    redshift: MaybeFloat = Free  # this could be optional
    inclination: MaybeFloat = Free
    fixed_metallicity: MaybeFloat = Just(0.)  # solar metallicity

    # Optional values
    # {Nothing | OptionalValue(Free) | OptionalValue(Just(<float>))}
    agn_eb_v: Optional = OptionalValue(Free)
    agn_torus_mass: Optional = OptionalValue(Free)


# ======================== Quasar Template Parameters =========================


class QuasarTemplateParams(ConfigClass):
    results_dir: str = 'results'
    quasar_data_loc: str = './data/quasar_template_shang.txt'
    torus_model_loc: str = './data/torus_model_with_inclination.dill'
    interpolated_quasar_loc: str = './data/quasar_template_interpolated.dill'

    def results_path(self, file: str) -> str:
        return os.path.join(self.results_dir, file)


# ====================== Extinction Template Parameters =======================


class ExtinctionTemplateParams(ConfigClass):
    interpolated_smc_extinction_loc: str \
                       = './data/interpolated_smc_extinction.dill'
    smc_data_loc: str = './data/smc_extinction_prevot_1984.dat'
    results_dir: str = 'results'

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
        # ,
        # '__main__': {  # if __name__ == '__main__'
        #     'handlers': ['file', 'console_debug'],
        #     'level': 'NOTSET',
        #     'propagate': True
        # },
    }
}


def configure_logging() -> None:
    """Performs one-time configuration of the root logger for the program.
    """
    dictConfig(logging_config)
    logging.info(
        '\n\n\n\n\n==================== New Run ===================\n\n')


# Utility classes -------------------------------------------------------------


class FreeParams(ConfigClass):
    def __init__(self, params: paramspace_t):
        self.raw_params: paramspace_t = params

        self.params: t.Tensor = t.empty((0, 2), dtype=t.float64)
        self.log: t.Tensor = t.empty((0, 1), dtype=t.bool)

        for p in params:
            setattr(self, p, params[p])
            self.log = t.cat(
                (self.log, t.tensor([[1 if p.startswith('log') else 0]],
                                    dtype=t.bool)))
            self.params = t.cat(
                (self.params, t.tensor([params[p]], dtype=t.float64)))

        # Remove unnecessary singleton dimension in mask
        self.log = self.log.squeeze(-1)

    def __len__(self) -> int:
        return len(self.raw_params)
