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
"""The main simulation class, containing methods to map from SPS parameters,
to photometry.
"""

import os
import tqdm
import h5py
import logging
import argparse
import torch as t
import numpy as np

from agnfinder import config as cfg
from agnfinder.simulation import utils
from agnfinder.types import FilterSet, Filters, paramspace_t
from agnfinder.prospector import Prospector


class Simulator(object):

    def __init__(self, args: argparse.Namespace, free_params: cfg.FreeParams):
        super(Simulator, self).__init__()

        self.hcube_sampled: bool = False
        self.has_forward_model: bool = False
        self.has_run: bool = False

        # Is this necessary?
        self.lims: paramspace_t = free_params.raw_params
        self.free_params = free_params
        self.dims: int = len(self.free_params)

        # Variables for saving results.
        rshift_min_string = f'{args.rshift_min:.4f}'.replace('.', 'p')
        rshift_max_string = f'{args.rshift_max:.4f}'.replace('.', 'p')
        self.save_name = 'photometry_simulation_{}n_z_{}_to_{}.hdf5'.format(
            args.n_samples, rshift_min_string, rshift_max_string
        )
        self.save_dir: str = args.save_dir
        self.save_loc: str = os.path.join(self.save_dir, self.save_name)

        self.n_samples: int = args.n_samples
        self.emulate_ssp: bool = args.emulate_ssp
        self.noise: bool = args.noise

        self.filters: FilterSet = self._str_to_filter(args.filters)
        self.output_dim = self.filters.dim

        self.rshift_range: tuple[float, float] = \
            (args.rshift_min, args.rshift_max)

        # The hypercube of (denormalised) galaxy parameters
        self.theta: t.Tensor
        logging.debug('Successfully initialised Simulator object')

    def sample_theta(self) -> None:
        """Generates a dataset via latin hypercube sampling."""
        # Use latin hypercube sampling to generate photometry from across the
        # parameter space.
        self.hcube = utils.get_unit_latin_hypercube(
            self.dims, self.n_samples
        )

        # Shift normalised redshift parameter to lie within the desired range.
        self.hcube[:, 0] = utils.shift_redshift_theta(
            self.hcube[:, 0], self.free_params.redshift, self.rshift_range
        )

        # Transform the unit-sampled point back to their correct ranges in the
        # parameter space (taking logs if needed).
        self.theta = utils.denormalise_theta(self.hcube, self.free_params)
        self.hcube_sampled = True
        logging.debug('Sampled galaxy parameters')

    def create_forward_model(self):
        """Initialises a Prospector problem, and obtains the forward model."""

        problem = Prospector(self.filters, self.emulate_ssp)

        # The calculate_sed method has side-effects, which includes updating
        # the problem.obs dict.
        problem.calculate_sed()
        self.phot_wavelengths = problem.obs['phot_wave']

        self.forward_model = problem.get_forward_model()
        self.has_forward_model = True
        logging.info('Created forward model')

    def run(self):
        """Run the sampling over all the galaxy parameters."""

        # Verify that we have all the required data / objects
        if not self.hcube_sampled:
            self.sample_theta()
        if not self.has_forward_model:
            self.create_forward_model()

        Y = np.zeros((self.n_samples, self.output_dim))
        for n in tqdm.tqdm(range(len(self.theta))):
            Y[n] = self.forward_model(self.theta[n].numpy())
        self.galaxy_photometry = Y
        self.has_run = True

    def save_samples(self):
        """Save the sampled parameter hypercube, and resulting photometry to
        disk as hdf5.

        Raises:
            RuntimeError: If sampling has not yet been run (nothing to save).
        """

        # If sampling has run, then we also have hcube and forward_model
        if not self.has_run:
            raise RuntimeError('Nothing to save')

        with h5py.File(self.save_loc, 'w') as f:
            grp = f.create_group('samples')
            ds_x = grp.create_dataset('theta', data=self.theta.numpy())
            # TODO does order matter?
            # If so, are free_params in the correct order?
            ds_x.attrs['columns'] = list(self.free_params.raw_params.keys())
            ds_x.attrs['description'] = 'Parameters used by simulator'

            ds_x_norm = grp.create_dataset('normalised_theta', data=self.hcube)
            ds_x_norm.attrs['description'] = \
                'Normalised parameters used by simulator'

            ds_y = grp.create_dataset(
                'simulated_y', data=self.galaxy_photometry)
            ds_y.attrs['description'] = 'Response of simulator'

            if self.phot_wavelengths is not None:
                ds_wavelengths = grp.create_dataset(
                    'wavelengths', data=self.phot_wavelengths)
                ds_wavelengths.attrs['description'] = \
                    'Observer wavelengths to visualise simulator photometry'

        logging.info(f'Saved samples to {self.save_loc}')

    def _str_to_filter(self, name: str) -> FilterSet:
        """Convenience method to convert filter name to filter.

        Args:
            name: The valid filter name

        Returns:
            FilterSet: The corresponding filter type

        Raises:
            ValueError: upon unrecognized filter name
        """
        if name == 'euclid':
            return Filters.Euclid
        elif name == 'reliable':
            return Filters.Reliable
        elif name == 'all':
            return Filters.All
        else:
            raise ValueError(f'Unrecognised filter name {name}')

if __name__ == '__main__':

    # Configure the root logger using the values in config.py:logging_config
    cfg.configure_logging()

    # Get the defaults from config.py
    sp = cfg.SamplingParams()
    fp = cfg.FreeParams()
    sps = cfg.SPSParams()

    parser = argparse.ArgumentParser(description='Find AGN')
    parser.add_argument(
            '--n_samples', default=sp.n_samples, type=int)
    parser.add_argument(
            '--z-min', dest='rshift_min', default=sp.redshift_min, type=float)
    parser.add_argument(
            '--z-max', dest='rshift_max', default=sp.redshift_max, type=float)
    parser.add_argument(
            '--save-dir', dest='save_dir', type=str, default=sp.save_dir)
    parser.add_argument(
            '--emulate-ssp', default=sps.emulate_ssp, action='store_true')
    parser.add_argument(
            '--noise', default=False, action='store_true')
    parser.add_argument(
            '--filters', dest='filters', type=str, default=sp.filters.value)
    args = parser.parse_args()

    logging.info(f'Free parameters are: {fp}')
    logging.info(f'Default sampling params are: {sp}')
    logging.info(f'Simulation arguments are: {args}')

    sim = Simulator(args, fp)

    # Latin hypercube sampling for the galaxy parameters.
    sim.sample_theta()

    # Create the forward model using Prospector
    sim.create_forward_model()

    # Simulate the photometry (automatically saves results to disk)
    sim.run()

    # Save the sample results to disk
    sim.save_samples()
