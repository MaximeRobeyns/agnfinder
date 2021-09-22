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
"""SED visualisations."""

import logging
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from sedpy.observate import Filter

from agnfinder.types import cpz_obs_dict_t, SEDComponents
from agnfinder.prospector import Prospector


CCOLOURS = {
    'galaxy': 'g',
    'unextincted_quasar': 'b',
    'extincted_quasar': 'b',
    'torus': 'orange',
    'net': 'k'
}


def plot_obs_photometry(obs: cpz_obs_dict_t):
    """plots observations, in observer frame
    """
    plt.errorbar(obs['phot_wave'], obs['maggies'], yerr=obs['maggies_unc'],
        label='Observed Photometry',
        marker='o', markersize=10, alpha=0.8, ls='', lw=3,
        ecolor='red', markerfacecolor='none', markeredgecolor='red',
        markeredgewidth=3)


def plot_filters(ax: plt.Axes, obs: cpz_obs_dict_t, ymin: float, ymax: float):
    """plots the filters used to generate the photometry
    """
    for f in obs['filters']:
        assert isinstance(f, Filter)
        w, t = f.wavelength.copy(), f.transmission.copy()
        t = t / t.max()
        t = 10**(0.2*(np.log10(ymax/ymin)))*t*ymin
        ax.loglog(w, t, lw=3, color='gray', alpha=0.7)


def visualise_obs(obs: cpz_obs_dict_t):
    """Visualise the photometric observations

    Args:
        obs: the observation dictionary
    """
    (xmin, xmax), (ymin, ymax) = get_bounds(obs)
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_obs_photometry(obs)
    plot_filters(ax, obs, ymin, ymax)
    prettify(fig, ax, xmin, xmax, ymin, ymax)


def prettify(fig: plt.Figure, ax: plt.Axes, xmin: float, xmax: float,
             ymin: float, ymax: float):
    """Set plot attributes to make it pretty.
    """
    ax.set_xlabel('Wavelength, A')
    ax.set_ylabel('Flux Denstiy, maggies')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    fig.legend(loc='upper center', fontsize=20)
    fig.tight_layout()


def component_to_band(component_list):
    # assumes all wavelengths are the same
    component_array = np.array(component_list)
    upper_limit = np.percentile(component_array, .9, axis=0)
    lower_limit = np.percentile(component_array, .1, axis=0)
    return upper_limit, lower_limit


def calculate_many_component(p: Prospector, theta_array: np.ndarray,
                             ax: plt.Axes = None):
    if ax == None:
        _, ax = plt.subplots(figsize=(16, 6))
    assert ax is not None
    all_components: list[SEDComponents] = []
    for theta in theta_array:
        _ = p.calculate_sed(theta=theta)
        all_components.append(p.get_components())

    # comp_names = ['wavelengths', 'galaxy', 'unextincted_quasar',
    #               'extincted_quasar', 'torus', 'net']
    comp_names = ['galaxy', 'extincted_quasar', 'torus', 'net']

    for cn in comp_names:
        upper, lower = component_to_band([c.__getattribute__(cn) for c in all_components])
        ax.loglog(all_components[0].wavelengths, upper, color=CCOLOURS[cn], alpha=0.3)
        ax.loglog(all_components[0].wavelengths, lower, color=CCOLOURS[cn], alpha=0.3)
        ax.fill_between(all_components[0].wavelengths, lower, upper, color=CCOLOURS[cn], alpha=0.3, label=cn)
    ax.set_xlabel('Wavelength (A), Source Frame')
    ax.set_ylabel('Flux Density (before Dimming)')


# Utility functions -----------------------------------------------------------


def get_bounds(obs: cpz_obs_dict_t, wspec: Optional[np.ndarray] = None,
               initial_spec: Optional[np.ndarray] = None
               ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Gets appropriate bounds on the figure (both x and y).

    Args:
        obs: observation dictionary from prospector class
        wspec: monotonically increasing sequence of x data points
        initial_spec: monotonically increasing sequence of y points

    Returns:
        tuple[tuple[float, float], tuple[float, float]]:
            (xmin, xmax), (ymin, ymax)

    Raises:
        ValueError: If `initial_spec` is not set while `wspec` is.
    """
    photometry_wavelengths = obs['phot_wave']

    xmin = np.min(photometry_wavelengths)*0.8
    xmax = np.max(photometry_wavelengths)/0.8

    if wspec is not None:  # interpolate sed to calculate y bounds
        if initial_spec is None:
            logging.error(f'initial_spec cannot be None if wspec is defined')
            raise ValueError('initial_spec not set')
        # evaluate wspec (x) vs. initial spec (y), along new x grid
        tmp = np.interp(np.linspace(xmin, xmax, 10000), wspec, initial_spec)
        ymin, ymax = tmp.min()*0.8, tmp.max()/0.4
    else:
        assert isinstance(obs['maggies'], np.ndarray)
        ymin, ymax = obs['maggies'].min()*0.4, obs['maggies'].max()/0.4

    return (xmin, xmax), (ymin, ymax)
