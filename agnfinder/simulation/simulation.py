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
""" The main simulation class, containing methods to map from SPS parameters,
to photometry.
"""

import argparse
import torch as t

from agnfinder import config as cfg
from agnfinder.simulation import utils
from agnfinder.types import paramspace_t
from agnfinder.prospector import Prospector


class Simulator(object):

    def __init__(self, args: argparse.Namespace, free_params: cfg.FreeParams):
        super(Simulator, self).__init__()

        self.hcube_sampled: bool = False
        self.has_forward_model: bool = False

        # Is this necessary?
        self.lims: paramspace_t = free_params.raw_params
        self.free_params = free_params  # TODO verify this line
        self.dims: int = len(self.free_params)

        rshift_min_string = f'{args.rshift_min:.4f}'.replace('.', 'p')
        rshift_max_string = f'{args.rshift_max:.4f}'.replace('.', 'p')
        self.save_name = 'photometry_simulation_{}n_z_{}_to_{}.hdf5'.format(
            args.n_samples, rshift_min_string, rshift_max_string
        )
        self.n_samples: int = args.n_samples
        self.save_dir: str = args.save_dir
        self.emulate_ssp: bool = args.emulate_ssp
        self.noise: bool = args.noise
        self.filters: str = args.filters
        self.rshift_range: tuple[float, float] = (
            args.rshift_min, args.rshift_max)

        self.theta: t.Tensor

    def sample_theta(self) -> None:
        """ Generates a dataset via latin hypercube sampling. """
        # Use latin hypercube sampling to generate photometry from across the
        # parameter space.
        hcube = utils.get_unit_latin_hypercube(
            self.dims, self.n_samples
        )

        # Shift normalised redshift parameter to lie within the desired range.
        hcube[:, 0] = utils.shift_redshift_theta(
            hcube[:, 0], self.free_params.redshift, self.rshift_range
        )

        # Transform the unit-sampled point back to their correct ranges in the
        # parameter space (taking logs if needed).
        self.galaxy_params = utils.denormalise_theta(hcube, self.free_params)
        self.hcube_sampled = True

    def create_forward_model(self):

        # TODO this is terrible form. Refactor?

        problem = Prospector(self.filters, self.emulate_ssp)

        # why is the result of this not used? Can we get rid of it?
        problem.calculate_sed()

        phot_wavelengths = problem.obs['phot_wave']

        self.forward_model = problem.forward_model

        self.has_forward_model = True
        raise NotImplementedError

    def run(self):
        """Run the sampling over all the galaxy parameters.
        """

        # Verify that we have all the required data / objects
        if not self.hcube_sampled:
            self.sample_theta()
        if not self.has_forward_model:
            self.create_forward_model()

        Y = np.zeros((n_samples, output_dim))
        for n in tqdm(range(len(self.galaxy_params))):
            Y[n] = self.forward_model(self.galaxy_params[n])
        return Y


if __name__ == '__main__':

    # Get the defaults from config.py
    sp = cfg.SamplingParams()
    fp = cfg.FreeParams

    parser = argparse.ArgumentParser(description='Find AGN')
    parser.add_argument(
            'n_samples', default=sp.n_samples, type=int)
    parser.add_argument(
            '--z-min', dest='rshift_min', default=sp.redshift_min, type=float)
    parser.add_argument(
            '--z-max', dest='rshift_max', default=sp.redshift_max, type=float)
    parser.add_argument(
            '--save-dir', dest='save_dir', type=str, default=sp.save_dir)
    parser.add_argument(
            '--emulate-ssp', default=sp.emulate_ssp, action='store_true')
    parser.add_argument(
            '--noise', default=False, action='store_true')
    parser.add_argument(
            '--filters', dest='filters', type=str, default=sp.filters)
    args = parser.parse_args()

    sim = Simulator(args, fp)

    # Latin hypercube sampling for the galaxy parameters.
    sim.sample_theta()

    # Create the forward model using Prospector
    sim.create_forward_model()

    # Simulate the photometry (automatically saves results to disk)
    sim.run()
