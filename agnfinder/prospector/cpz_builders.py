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

import logging

from typing import Callable
from prospect.models import priors
from prospect.models import templates
from prospect.models.sedmodel import SedModel

import agnfinder.config as cfg
from agnfinder.types import pdict_t

# Closures to use either Optional:Nothing, MaybeFloat: Fixed, or
# MaybeFloat:Just(val) values.
def _use_nothing(name: str) -> Callable[[], None]:
    def f() -> None:
        logging.warning(f'Not modelling {name}')
    return f

def _use_free(name: str, defaults: pdict_t = {}) -> Callable[[], pdict_t]:
    def f() -> pdict_t:
        logging.info(f'Using free {name}')
        return defaults | {'isfree': True}
    return f

def _use_float(name: str, defaults: pdict_t = {}) -> Callable[[float], pdict_t]:
    def f(val: float) -> pdict_t:
        logging.info(f'Using fixed {name} of {val}')
        return defaults | {'isfree': False, 'init': val}
    return f


def build_model(args: cfg.CPzModelParams) -> SedModel:
    """Build a SedModel object using the provided parameters.

    Args:
        args: Defined in config.py

    Returns:
        SedModel: The prospector SED model.
    """

    logging.debug("redshift: ",
            "free" if args.redshift is None else args.redshift)
    logging.debug("fixed_metallicity: ", args.fixed_metallicity)
    logging.debug("dust", args.dust)
    logging.debug("agn_mass", args.agn_mass)
    logging.debug("agn_eb_v", args.agn_eb_v)
    logging.debug("agn_torus_mass", args.agn_torus_mass)
    logging.debug("igm_absorbtion", args.igm_absorbtion)

    # Get a copy of one of the pre-packaged model set dictionaries.
    model_params = templates.TemplateLibrary['parametric_sfh']
    model_params['dust_type'] = {'N': 1, 'isfree': False, 'init': 2}

    while True:  # allows to skip to end if args.model_agn == False

        # metallicity parameter
        model_params['logzsol'] |= args.fixed_metallicity.use(
                _use_free('metallicity'),
                _use_float('metallicity'))

        # redshift param
        model_params['zred'] |= args.redshift.use(
                _use_free('redshift'),
                _use_float('redshift'))

        # dust parameter
        if args.dust:
            logging.info('Including dust emissions fixed parameters')
            model_params |= templates.TemplateLibrary['dust_emission']
        else:
            logging.warning('Not using dust emission')

        # igm absorbtion parameter
        if args.igm_absorbtion:
            logging.info('Using fixed IGM absorbtion (0.1 by default)')
            model_params |= templates.TemplateLibrary['igm']
        else:
            logging.warning('Not using IGM absorbtion')


        if not args.model_agn:
            logging.warning('AGN Not being modelled')
            break

        # AGN parameters -------------------------------------------------------

        # agn mass parameter
        model_params['agn_mass'] |= args.agn_mass.use(
                _use_free('agn mass', {'N': 1, 'init': 1,
                          'prior': priors.LogUniform(mini=1e-7, maxi=15)}),
                _use_float('agn_mass', {'N': 1}))

        model_params['agn_eb_v'] |= args.agn_eb_v.use(
                _use_nothing('AGN distk'),
                _use_free('agn_eb_v', {'N': 1, 'init': 0.1, 'units': '',
                          'prior': priors.TopHat(mini=0., maxi=0.5)}),
                _use_float('agn_eb_v', {'N': 1, 'units': '',
                          'prior': priors.TopHat(mini=0., maxi=0.5)}))

        model_params['agn_torus_mass'] |= args.agn_torus_mass.use(
                _use_nothing('AGN torus'),
                _use_free('obscured torus', {'N': 1, 'init': .1, 'units': '',
                          'prior': priors.LogUniform(mini=1e-7, maxi=15)}),
                _use_float('obscured torus', {'N': 1, 'units': '',
                          'prior': priors.LogUniform(mini=1e-7, maxi=15)}))

        model_params['inclination'] |= args.inclination.use(
                _use_free('inclination', {'N': 1, 'init': 60, 'units': '',
                          'prior': priors.TopHat(mini=0., maxi=90.)}),
                _use_float('inclination', {'N': 1, 'units': '',
                           'prior': priors.TopHat(mini=0., maxi=90.)}))
        break

    return SedModel(model_params)
