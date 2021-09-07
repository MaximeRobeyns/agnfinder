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
import numpy as np

from multiprocessing import Pool

from agnfinder import config as cfg
from agnfinder.simulation import utils
from agnfinder.types import Tensor, FilterSet, Filters, paramspace_t
from agnfinder.prospector import Prospector


class Simulator(object):

    def __init__(self, rshift_min: float, rshift_max: float,
                 worker_idx: int = 0) -> None:
        """Initialise the simulator class.

        The majority of the arguments are obtained from the configuration file.

        Args:
            rshift_min: Minimum redshift value (must be non-negative)
            rshift_max: Maximum redshift value for this cube
            worker_idx: index of this process, if in worker pool
        """
        super(Simulator, self).__init__()

        free_params = cfg.FreeParams()
        s_params = cfg.SamplingParams()
        sps_params = cfg.SPSParams()

        n_samples = int(s_params.n_samples / s_params.concurrency)

        self.hcube_sampled: bool = False
        self.has_forward_model: bool = False
        self.has_run: bool = False

        # Is this necessary?
        self.lims: paramspace_t = free_params.raw_params
        self.free_params = free_params
        self.dims: int = len(self.free_params)

        # Variables for saving results.
        rshift_min_string = f'{rshift_min:.4f}'.replace('.', 'p')
        rshift_max_string = f'{rshift_max:.4f}'.replace('.', 'p')
        self.save_name = 'photometry_simulation_{}n_z_{}_to_{}_w_{}.hdf5'.format(
            n_samples, rshift_min_string, rshift_max_string, worker_idx
        )
        self.widx = worker_idx
        self.save_dir: str = os.path.join(s_params.save_dir, "partials")
        self.save_loc: str = os.path.join(self.save_dir, self.save_name)

        self.n_samples: int = n_samples
        self.emulate_ssp: bool = sps_params.emulate_ssp
        self.noise: bool = s_params.noise

        self.filters: FilterSet = s_params.filters
        self.output_dim = self.filters.dim

        self.rshift_range: tuple[float, float] = (rshift_min, rshift_max)

        # The hypercube of (denormalised) galaxy parameters
        self.theta: Tensor
        if self.widx == 0:
            logging.info(f'Initialised simulator')

    def sample_theta(self) -> None:
        """Generates a dataset via latin hypercube sampling."""
        if self.widx == 0:
            logging.info((f'Drawing {self.n_samples} samples from '
                          f'{self.dims}-dimensional space...'))
        # Use latin hypercube sampling to generate photometry from across the
        # parameter space.
        self.hcube = utils.get_unit_latin_hypercube(
            self.dims, self.n_samples
        )
        if self.widx == 0:
            logging.info(f'Completed Latin-hypercube sampling.')

        # Shift normalised redshift parameter to lie within the desired range.
        self.hcube[:, 0] = utils.shift_redshift_theta(
            self.hcube[:, 0], self.free_params.redshift, self.rshift_range
        )

        # Transform the unit-sampled point back to their correct ranges in the
        # parameter space (taking logs if needed).
        self.theta = utils.denormalise_theta(self.hcube, self.free_params)
        self.hcube_sampled = True
        if self.widx == 0:
            logging.debug(f'Sampled galaxy parameters')

    def create_forward_model(self):
        """Initialises a Prospector problem, and obtains the forward model."""

        problem = Prospector(self.filters, self.emulate_ssp)

        # The calculate_sed method has side-effects, which includes updating
        # the problem.obs dict.
        problem.calculate_sed()
        self.phot_wavelengths = problem.obs['phot_wave']

        self.forward_model = problem.get_forward_model()
        self.has_forward_model = True
        if self.widx == 0:
            logging.info(f'Created forward model')

    def run(self):
        """Run the sampling over all the galaxy parameters."""

        # Verify that we have all the required data / objects
        if not self.hcube_sampled:
            self.sample_theta()
        if not self.has_forward_model:
            self.create_forward_model()

        Y = np.zeros((self.n_samples, self.output_dim))
        if self.widx == 0:
            for n in tqdm.tqdm(range(len(self.theta)), self.widx):
                Y[n] = self.forward_model(self.theta[n].numpy())
        else:
            for n in range(len(self.theta)):
                Y[n] = self.forward_model(self.theta[n].numpy())
        self.galaxy_photometry = Y
        self.has_run = True
        logging.info(f'Worker {self.widx} finished simulation run')

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
            ds_x.attrs['columns'] = self.free_params.raw_members
            ds_x.attrs['description'] = 'Parameters used by simulator'

            # TODO should we be saving denormalised theta here?
            ds_x_norm = grp.create_dataset('normalised_theta', data=self.hcube)
            ds_x_norm.attrs['columns'] = self.free_params.raw_members
            ds_x_norm.attrs['description'] = \
                'Normalised parameters used by simulator'

            ds_y = grp.create_dataset(
                'simulated_y', data=self.galaxy_photometry)
            ds_y.attrs['description'] = 'Response of simulator'

            ds_wavelengths = grp.create_dataset(
                'wavelengths', data=self.phot_wavelengths)
            ds_wavelengths.attrs['description'] = \
                'Observer wavelengths to visualise simulator photometry'
        logging.debug(f'Saved partial samples to {self.save_loc}')

    def _str_to_filter(self, name: str) -> FilterSet:
        """Convenience method to convert filter name to filter.
        NOTE: this is no longer used (since cli args no longer accepted)

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


def work_func(zmin: float, zmax: float, worker_idx: int) -> None:
    """The entrypoint for the forked process in the worker pool to run the
    sampling on a sub-cube.

    Args:
        zmin: minimum redshift for this sub-cube
        zmax: maximum redshift for this sub-cube
        worker_idx: unique job index
    """

    logging.info(f'Beginning worker process: {worker_idx}')
    sim = Simulator(rshift_min=zmin, rshift_max=zmax, worker_idx=worker_idx)

    # Latin hypercube sampling for the galaxy parameters.
    sim.sample_theta()

    # Create the forward model using Prospector
    sim.create_forward_model()

    # Simulate the photometry (automatically saves results to disk)
    sim.run()

    # Save the sample results to disk
    sim.save_samples()


if __name__ == '__main__':

    # Configure the root logger using the values in config.py:logging_config
    cfg.configure_logging()

    # Get the defaults from config.py
    sp = cfg.SamplingParams()

    # Ensure temporary save location is created and empty
    utils.ensure_partials_dir(sp.save_dir)

    # prepare job list
    samples_per_cube: int = int(sp.n_samples / sp.concurrency)
    inc: float = (sp.redshift_max - sp.redshift_min) / sp.concurrency
    zrange: list[float] = [i  * inc for i in range(sp.concurrency+1)]
    zlims: list[tuple[float, float]] = list(zip(zrange[:-1], zrange[1:]))
    p_args: list[tuple[float, float, int]] = \
        [(a, b, i) for i, (a, b) in enumerate(zlims)]

    # assert split sampling space evenly between processes
    assert len(zlims) == sp.concurrency

    # run the actual simulation (in worker pool)
    with Pool(sp.concurrency) as pool:
        pool.starmap(work_func, p_args)

    # join and save samples
    utils.join_partial_samples(sp)

