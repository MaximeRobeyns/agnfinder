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
"""Builders for 'classification-aided photometric redshfit' estimation
module"""

import logging
import numpy as np

from typing import Callable

from prospect.utils.obsutils import fix_obs
from prospect.models import priors
from prospect.models import templates
from prospect.models.sedmodel import SedModel
from prospect.sources import CSPSpecBasis

from agnfinder.config import CPzParams, SPSParams
from agnfinder.types import FilterSet, cpz_obs_dict_t, pdict_t
from agnfinder.prospector import load_photometry
from agnfinder.prospector.csp_classes import CSPSpecBasisAGN, CSPSpecBasisNoEm


def build_cpz_obs(filter_selection: FilterSet, catalogue_loc: str = ""
                 ) -> cpz_obs_dict_t:
    """Build a dictionary of photometry (and maybe eventually spectra).

    Args:
        filter_selection: the SPS filter selection to use
        catalogue_loc: an optional string giving the catalogue location, to
            replace the dummy galaxy with a real one.

    Returns:
        None: A dictionary of observational data to use in the fit.
    """

    # load galaxy only if catalogue location is provided
    galaxy = load_photometry.load_galaxy(catalogue_loc) if catalogue_loc != "" \
             else None

    obs: cpz_obs_dict_t = {}
    if galaxy is not None:
        f, m, mu = load_photometry.load_galaxy_for_prospector(
            galaxy, filter_selection)
    else:
        f, m, mu = load_photometry.load_dummy_galaxy(filter_selection)
    obs['filters'] = f
    obs['maggies'] = m
    obs['maggies_unc'] = mu

    # This mask tells us which flux values to consider in the likelihood.
    # NOTE: the mask is _True_ for values that you _want_ to fit.
    obs['phot_mask'] = np.array([True for _ in obs['filters']])

    # This is an array of the effective wavelengths for each of the filters.
    # Unnecessary, but useful for plotting so it's stored here for convenience
    obs['phot_wave'] = np.array([f.wave_effective for f in obs['filters']])

    # Since we don't have a spectrum, we set some required elements of the obs
    # directory to None. (This would be a vector of vacuum wavelengths in
    # angstroms.) NOTE: Could use the SDSS spectra here for truth label fitting
    # This is inelegant wrt. the types of cpz_obs_dict_t but since this is a
    # Prospector API, it's not trivial to get aroud...
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['unc'] = None
    obs['mask'] = None

    obs = fix_obs(obs)

    return obs


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


def build_model(args: CPzParams) -> SedModel:
    """Build a SedModel object using the provided parameters.

    Args:
        args: CPz parameters, defined in config.py

    Returns:
        SedModel: The prospector SED model.
    """

    logging.info(f'Building SedModel model')

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
            logging.warning('AGN not being modelled')
            break

        # AGN parameters -------------------------------------------------------

        # agn mass parameter
        model_params['agn_mass'] = args.agn_mass.use(
                _use_free('agn mass', {'N': 1, 'init': 1.,
                          'prior': priors.LogUniform(mini=1e-7, maxi=15)}),
                _use_float('agn_mass', {'N': 1}))

        model_params['agn_eb_v'] = args.agn_eb_v.use(
                _use_nothing('AGN extinction'),
                _use_free('agn_eb_v', {'N': 1, 'init': 0.1, 'units': '',
                          'prior': priors.TopHat(mini=0., maxi=0.5)}),
                _use_float('agn_eb_v', {'N': 1, 'units': '',
                          'prior': priors.TopHat(mini=0., maxi=0.5)}))

        model_params['agn_torus_mass'] = args.agn_torus_mass.use(
                _use_nothing('AGN torus'),
                _use_free('obscured torus', {'N': 1, 'init': .1, 'units': '',
                          'prior': priors.LogUniform(mini=1e-7, maxi=15)}),
                _use_float('obscured torus', {'N': 1, 'units': '',
                          'prior': priors.LogUniform(mini=1e-7, maxi=15)}))

        model_params['inclination'] = args.inclination.use(
                _use_free('inclination', {'N': 1, 'init': 60., 'units': '',
                          'prior': priors.TopHat(mini=0., maxi=90.)}),
                _use_float('inclination', {'N': 1, 'units': '',
                           'prior': priors.TopHat(mini=0., maxi=90.)}))
        break

    return SedModel(model_params)


def build_sps(cpz_params: CPzParams, sps_params: SPSParams) -> CSPSpecBasis:
    """Build stellar population synthesis model

    Args:
        args: CPz parameter configuration object
        emulate_ssp: Whether to emulate the ssp
        zcontinuous: A value of 1 ensures that we use interpolation between
        SSPs to have a continuous metallicity parameter (`logzsol`). See pyFSPS
        documentation for more details.

    Returns:
        CSPSpecBasis: Instance of a CSPSpecBasis class
    """

    logging.info(f'Building SPS model')

    if cpz_params.model_agn:
        logging.info('Building custom CSPSpecBasisAGN.')
        sps = CSPSpecBasisAGN(
            zcontinuous=sps_params.zcontinuous,
            reserved_params=sps_params.reserved_params,
            vactoair_flag=sps_params.vactoair_flag,
            compute_vega_mags=sps_params.compute_vega_mags,
            emulate_ssp=sps_params.emulate_ssp,
            agn_mass=cpz_params.agn_mass.value,
            agn_eb_v=cpz_params.agn_eb_v.value,
            agn_torus_mass=cpz_params.agn_torus_mass.value,
            inclination=cpz_params.inclination.value)
    else:
        logging.info('Building standard CSPSpec')
        sps = CSPSpecBasisNoEm(zcontinuous=sps_params.zcontinuous)

    return sps
