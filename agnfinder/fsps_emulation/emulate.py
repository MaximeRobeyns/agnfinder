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
"""GP emulator for FSPS forward model.

Note that this is not really used, but kept in for reference.
"""

import os
import pickle
import numpy as np
import GPy

class SKLearnEmulator():
    def __init__(self, model_loc: str):
        with open(model_loc, 'rb') as f:
            self._model = pickle.load(f)

    def __call__(self, params: np.ndarray) -> np.ndarray:
        return self._model.predict(params.reshape(1, -1))

class GPEmulator():

    def __init__(self, gp_model_loc: str, pca_model_loc: str):

        # Assume that all these files are in the same directory as gp_model_loc.
        self.model_loc = os.path.dirname(gp_model_loc)
        self.x_mean = self._loadnp('x_mean.txt')
        self.x_mult = self._loadnp('x_mult.txt')
        self.y_mean = self._loadnp('y_mean.txt')
        self.y_mult = self._loadnp('y_mult.txt')

        with open(pca_model_loc, 'rb') as f:
            self._pca_model = pickle.load(f)

        self._gp_model = GPy.models.GPRegression.load_model(gp_model_loc)

    def _loadnp(self, path: str) -> np.ndarray:
        return np.loadtxt(os.path.join(self.model_loc, path))

    def emulator(self, params: np.ndarray) -> np.ndarray:
        """Emulates new spectra from physical parameters.

        Args:
            params: Physical parameters; 1D vector of tau, tage and dust.

        Returns:
            np.ndarray: Emulated target (Y)
        """
        # normalise the parameters
        params = (params - self.x_mean)/self.x_mult

        # weight predictions
        params = np.expand_dims(params, axis=0)
        pred_weights = self._gp_model.predict(params)[0]

        # inverse PCA (pred_weights * basis + mean)
        reconstructed = self._pca_model.inverse_transform(pred_weights)
        # denormalise
        return 10**((reconstructed[0]*self.y_mult) + self.y_mean)

    def __call__(self, params: np.ndarray) -> np.ndarray:
        return self.emulator(params)
