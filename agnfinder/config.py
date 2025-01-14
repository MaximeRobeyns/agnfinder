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
import time
import typing
import torch as t
import torch.nn as nn
import logging
from typing import Any, Union, Type
from logging.config import dictConfig

# TODO put these in the inference module's __init__ file
# or, extend the 'enum' for inference_procedure to avoid having to
# import these modules at all
import agnfinder.inference as inference
import agnfinder.inference.san as san
import agnfinder.inference.cvae as cvae
import agnfinder.inference.made as made
# mcmc has a Prospector dependency.
import agnfinder.inference.mcmc_util as mcmc_util

from agnfinder.inference.utils import Squareplus
from agnfinder.types import FreeParameters, ConfigClass, arch_t, \
        MaybeFloat, Free, Just, \
        Optional, OptionalValue, Nothing, \
        FilterSet, Filters, \
        MCMCMethod, EMCEE, Dynesty, UltraNest
from agnfinder.inference.inference import model_t
from agnfinder.inference.utils import get_colours_length


# ============================= Free Parameters ===============================


class FreeParams(FreeParameters):
    """Note: Keys prefixed by 'log_*' are exponentiated later"""

    # redshift: tuple[float, float] = (0., 6.)
    redshift: tuple[float, float] = (0., 6.)
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

    column_order: list[str] = ['redshift', 'log_mass', 'dust2', 'tage',
            'log_tau', 'log_agn_mass', 'agn_eb_v', 'log_agn_torus_mass',
            'inclination']


# =========================== Sampling Parameters =============================


# These defaults can be overridden by command line arguments when invoking
# agnfinder/simulation/simulation.py (run with --help flag to see options)
class SamplingParams(ConfigClass):
    # n_samples: int = 40000000
    n_samples: int = 10000
    concurrency: int = 6  # set this to os.cpu_count() (or slightly less)
    redshift_min: float = 0.
    redshift_max: float = 6.
    save_dir: str = './data/cubes/des_log_z_sample'
    noise: bool = False
    filters: FilterSet = Filters.DES  # {Euclid, DES, Reliable, All}
    shuffle: bool = True  # whether to shuffle final samples


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
    vactoair_flag: bool = False
    compute_vega_mags: bool = False
    reserved_params: list[str] = ['zred', 'sigma_smooth']

    # Outdated parameter: allows to emulate FSPS simple stellar population
    # using e.g. a GP (or some other function approximator).
    emulate_ssp: bool = False
    # catalogue_loc: typing.Optional[str] = "./data/cpz_paper_sample_week3.parquet"
    catalogue_loc: typing.Optional[str] = None


# ======================== Quasar Template Parameters =========================


class QuasarTemplateParams(ConfigClass):
    results_dir: str = 'results'

    # quasar parameters
    recreate_quasar_template: bool = False
    quasar_data_loc: str = './data/quasar_template_shang.txt'
    interpolated_quasar_loc: str = './data/quasar_template_interpolated.dill'

    # dusty torus model paramteres
    recreate_torus_template: bool = False
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
    recreate_extinction_template: bool = False
    interpolated_smc_extinction_loc: str \
                       = './data/interpolated_smc_extinction.dill'
    smc_data_loc: str = './data/smc_extinction_prevot_1984.dat'

    def results_path(self, file: str) -> str:
        return os.path.join(self.results_dir, file)


# ======================= Inference (main) Parameters =========================


class InferenceParams(inference.InferenceParams):
    # The model to use for estimating PDFs
    model: model_t = san.SAN
    logging_frequency: int = 10000

    # Training Inference Models -----------------------------------------------
    # Note: not all methods (e.g. MCMC) will need this
    split_ratio: float = 0.8  # train / test split ratio
    # If you do not have this dataset, run `make sim`
    # If you update SamplingParams, you will need to change this file path!
    dataset_loc: str = './data/cubes/photometry_simulation_100000n_z_0p0000_to_1p0000.hdf5'
    retrain_model: bool = False  # don't re-train an identical (existing) model
    use_existing_checkpoints: bool = True
    overwrite_results: bool = False  # if retrain_model, don't overwrite old one
    ident: str = ''  # identifier for this training run

    # Predicting PDFs ---------------------------------------------------------
    # The catalogue of real observations
    catalogue_loc: str = './data/DES_VIDEO_v1.0.1.fits'
    filters: FilterSet = Filters.DES  # {Euclid, DES, Reliable, All}

    # MCMC 'gold standard' comparison -----------------------------------------
    mcmc_method: Type[MCMCMethod] = UltraNest # {EMCEE, Dynesty, UltraNest}
    mcmc_concurrency: int = 1  # number of galaxies to process concurrently
    mcmc_galaxies: int = 1  # number of galaxies to infer (-1 = all)



# ======================= Inference (MCMC) Parameters =========================

# some base parameters common to all MCMC procedures

class MCMCBase(mcmc_util.MCMCParams):

    cond_dim: int = InferenceParams().filters.dim
    data_dim: int = 9  # y; len(FreeParameters()); dimensions of physical params
    filters: FilterSet = InferenceParams.filters  # {Euclid, DES, Reliable, All}
    catalogue_loc: str = InferenceParams.catalogue_loc

# Dynesty

class DynestyParams(mcmc_util.DynestyParams, MCMCBase):
    method: str = 'rwalk'
    bound: str = 'multi'  # bounding method TODO make enum
    bootstrap: int = 0  # TODO make bool?
    sample: str = 'unif'  # sampling method TODO make enum
    nlive_init: int = 400
    nlive_batch: int = 200
    weight_kwargs: dict[str, Any] = {"pfrac": 1.0}
    stop_kwargs: dict[str, Any] = {"post_thresh": 0.1}
    dlogz_init: float = 0.05
    posterior_thresh: float = 0.05
    maxcall: int = 200
    # maxcall: int = int(1e7)

    optimize: bool = False
    nmin: int = 10
    min_method: str = ''  # 'lm' | 'powell'


# EMCEE


class EMCEEParams(mcmc_util.EMCEEParams, MCMCBase):
    nwalkers: int = 128
    nburn: list[int] = [512]
    niter: int = 10000
    interval: float = 0.25
    initial_disp: float = 0.1

    optimize: bool = True
    nmin: int = 10
    min_method: str = 'powell'  # 'lm' | 'powell'


# UltraNest


class UltraNestParams(mcmc_util.UltraNestParams, MCMCBase):
    """See inference/mcmc_utils.py for more about these parameters.
    Better yet, consult the UltraNest documentation:
    https://johannesbuchner.github.io/UltraNest/ultranest.html#module-ultranest.integrator
    """
    ndraw_min: int = 128
    ndraw_max: int = 65536

    # run specific parameters -------------------------------------------------

    update_interval_volume_fraction: float = 0.8
    show_status: bool = True
    viz_callback: Any = 'auto' # set to False for quiet output
    dlogz: float = 0.5
    dKL: float = 0.5
    frac_remain: float = 1e-1
    Lepsilon: float = 0.001
    min_ess: int = 2000
    max_iters: typing.Optional[int] = None
    max_ncalls: typing.Optional[int] = None
    max_num_improvement_loops: int = -1
    min_num_live_points: int = 400
    cluster_num_live_points: int = 40


# ======================= Inference (CVAE) Parameters =========================


class CVAEParams(cvae.CVAEParams):

    epochs: int = 10
    batch_size: int = 32
    dtype: t.dtype = t.float64

    # Use get_colours_length(cond_dim) if using colours.
    cond_dim: int = InferenceParams().filters.dim

    data_dim: int = 9  # y; len(FreeParameters()); dimensions of physical params
    latent_dim: int = 4  # z
    adam_lr: float = 1e-3

    # (conditional) Gaussian prior network p_{theta}(z | x)
    # prior = inference.StandardGaussianPrior
    prior: Type[cvae.CVAEPrior] = cvae.FactorisedGaussianPrior
    # prior_arch = None
    prior_arch: arch_t = arch_t(
        layer_sizes=[cond_dim, 16],
        activations=nn.SiLU(),
        head_sizes=[latent_dim, latent_dim],
        head_activations=[None, Squareplus(0.8)],
        batch_norm=True)

    # Gaussian recognition model q_{phi}(z | y, x)
    encoder: Type[cvae.CVAEEnc] = cvae.FactorisedGaussianEncoder
    enc_arch: arch_t = arch_t(
        layer_sizes=[data_dim + cond_dim, 16],
        activations=nn.SiLU(),
        head_sizes=[latent_dim, latent_dim], # mean and log_std
        head_activations=[None, Squareplus(0.2)],
        batch_norm=True)


    # Gaussian generator network arch: p_{theta}(y | z, x)
    decoder: Type[cvae.CVAEDec] = cvae.FactorisedGaussianDecoder
    dec_arch: arch_t = arch_t(
        layer_sizes=[latent_dim + cond_dim, 32, 16],
        head_sizes=[data_dim, data_dim],
        activations=nn.SiLU(),
        head_activations=[None, Squareplus(0.8)],
        batch_norm=True)


# ======================= Inference (MADE) Parameters =========================


class MADEParams(made.MADEParams):
    epochs: int = 20
    batch_size: int = 1024
    dtype: t.dtype = t.float32

    # Use get_colours_length(cond_dim) if using colours.
    cond_dim: int = InferenceParams().filters.dim

    data_dim: int = 9  # y; dimensions of physical parameters to be estimated
    # TODO make much larger and deeper
    hidden_sizes: list[int] = [4, 8]

    likelihood: Type[made.MADE_Likelihood] = made.Gaussian
    likelihood_kwargs: typing.Optional[dict[str, Any]] = None

    # number of different orderings for order / connection agnostic training
    num_masks: int = 16

    # How many samples of connectivity / masks to average parameters over during
    # inference
    samples: int = 4

    natural_ordering: bool = False


# ======================== Inference (SAN) Parameters =========================


class SANParams(san.SANParams):
    epochs: int = 20
    batch_size: int = 1024
    dtype: t.dtype = t.float32

    # Use get_colours_length(cond_dim) if using colours.
    cond_dim: int = InferenceParams().filters.dim

    data_dim: int = 9  # dimensions of data of interest (e.g. physical params)
    module_shape: list[int] = [512, 512]  # shape of the network 'modules'
    sequence_features: int = 8  # features passed between sequential blocks
    likelihood: Type[san.SAN_Likelihood] = san.MoG
    likelihood_kwargs: typing.Optional[dict[str, Any]] = {
            'K': 10, 'mult_eps': 1e-4, 'abs_eps': 1e-4}
    batch_norm: bool = True  # use batch normalisation in network?
    opt_lr: float = 1e-4

    # Simple alternative Gaussian likelihood (retained for reference)
    # likelihood: Type[san.SAN_Likelihood] = san.Gaussian
    # likelihood_kwargs = None


# =========================== Logging Parameters ==============================

# logging levels:
# CRITICAL (50) > ERROR (40) > WARNING (30) > INFO (20) > DEBUG (10) > NOTSET

class LoggingParams(ConfigClass):
    file_loc: str = './logs.txt'

    # If any of these levels are NOTSET, then the corresponding logging handler
    # will not be used.
    file_level: int = logging.INFO
    debug_level: int = logging.NOTSET  # logging.DEBUG
    console_level: int = logging.INFO


# ----------------------------------------------------------------------------
# Other logging configurations (you shouldn't have to change these).


def get_logging_config(p: LoggingParams) -> dict[str, Any]:

    handlers = []
    if p.file_level > logging.NOTSET:
        handlers.append('file')
    if p.debug_level > logging.NOTSET:
        handlers.append('console_debug')
    if p.console_level > logging.NOTSET:
        handlers.append('console')

    console_stream = 'ext://sys.stdout'
    if p.console_level > 20:
        console_stream = 'ext://sys.stderr'

    return {
        'version': 1,
        'disable_existing_loggers': False,
        # see: https://docs.python.org/3/library/logging.html#logrecord-attributes
        'formatters': {
            'standard': {
                'format': '[%(levelname)s] %(message)s'
            },
            'debug': {
                'format': '[Dbg: %(asctime)s %(levelname)s] %(message)s (in %(filename)s:%(lineno)d)'
            },
        },
        'handlers': {
            # only log errors out to the console
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': p.console_level,
                'stream': console_stream
            },
            'console_debug': {
                'class': 'logging.StreamHandler',
                'formatter': 'debug',
                'level': p.debug_level,
                # print everything to stdout
                'stream': 'ext://sys.stdout'
            },
            # Output more information to a file for post-hoc analysis
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'standard',
                'level': p.file_level,
                'filename': p.file_loc,
                'mode': 'a',
                'encoding': 'utf-8',
                'maxBytes': 500000,
                'backupCount': 4
            }
        },
        'loggers': {
            '': {  # root logger
                'handlers': handlers,
                'level': 'NOTSET',
                'propagate': True
            }
        }
    }


# Utility -------------------------------------------------------------


def configure_logging(console_level: Union[int, None] = None,
                      debug_level: Union[int, None] = None,
                      file_level: Union[int, None] = None,
                      file_loc: Union[str, None] = None):
    """Performs a one-time configuration of the root logger for the program.

    Note that all the arguments are optional, and if omitted the default values
    in config.py will be used.

    Args:
        console_level: level at which to output to stderr (e.g. logging.ERROR)
        debug_level: level at which to output to stdout (e.g. logging.DEBUG)
        file_level: level at which to output to log file (e.g. LOGGING.INFO)
        file_loc: location of the log file

    Example:
        >>> configure_logging(debug_level=logging.DEBUG)
    """
    lp = LoggingParams()  # default logging parameters
    if console_level is not None:
        lp.console_level = console_level
    if debug_level is not None:
        lp.debug_level = debug_level
    if file_level is not None:
        lp.file_level = file_level
    if file_loc is not None:
        lp.file_loc = file_loc
    dictConfig(get_logging_config(lp))
    logging.info(
        f'\n\n{79*"~"}\n\n\tAGNFinder\n\t{time.ctime()}\n\n{79*"~"}\n\n')
