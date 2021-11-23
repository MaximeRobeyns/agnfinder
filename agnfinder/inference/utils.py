# AGNfinder: Detect AGN from photometry in XXL data.
#
# Copyright (C) 2021 Maxime Robeyns <maximerobeyns@gmail.com>
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
"""Utility functions relating to inference."""

import os
import sys
import h5py
import logging
import warnings
import torch as t
import torch.nn as nn
import numpy as np

from torchvision import datasets, transforms
from typing import Any, Callable, Optional, Union
from torch.utils.data import Dataset, DataLoader, random_split
from agnfinder.types import Tensor, tensor_like, column_order


class InMemoryGalaxyDataset(Dataset):
    """This dataloader loads one or more hdf5 files containing generated
    (theta, photometry) pairs into (system) memory simultaneously.
    """

    def __init__(self, path: str,
                 normalise_phot: Optional[Callable[[Any], Any]] = None,
                 transforms: list[Callable[[Any], Any]] = [],
                 x_transforms: list[Callable[[Any], Any]] = [],
                 y_transforms: list[Callable[[Any], Any]] = []):
        """Load the galaxy dataset

        Args:
            path: either the path of a hdf5 file or directory containing hdf5
                files, with (theta, photometry) pairs, output from a simulation
                run.
            transforms: any transformations to apply to the data now
            x_transforms: photometry specific transforms
            y_transforms: parameter specific transforms
        """

        self.transforms = transforms
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        files = []

        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith(".hdf5"):
                    files.append(os.path.join(path, file))
            logging.info(f'Loading {len(files)} files into dataset.')
        elif os.path.isfile(path):
            files = [path]
            logging.info(f'Loading 1 file into dataset.')
        else:
            logging.error((f'Provided path ({path}) is neither a directory or '
                           f'path to hdf5 file.'))
            sys.exit()

        xs, ys = self._get_x_y_from_file(files[0])

        for f in files[1:]:
            tmp_xs, tmp_ys = self._get_x_y_from_file(f)
            xs = np.concatenate((xs, tmp_xs), 0)
            ys = np.concatenate((ys, tmp_ys), 0)

        if normalise_phot is not None:
            xs = normalise_phot(xs)

        # eagerly compute transformations (instead of during __getitem__)
        for xtr in self.x_transforms:
            xs = xtr(xs)
        for ytr in self.y_transforms:
            ys = ytr(ys)

        self._x_dim, self._y_dim = xs.shape[-1], ys.shape[-1]
        self.dataset = np.concatenate((xs, ys), -1)

        logging.info(f'Galaxy dataset loaded.')

    def _get_x_y_from_file(self, file: str) -> tuple[np.ndarray, np.ndarray]:
        assert os.path.exists(file)
        f = h5py.File(file, 'r')
        samples = f.get('samples')
        assert isinstance(samples, h5py.Group)

        # photometry
        xs = np.array(samples.get('simulated_y'))

        # physical parameters
        norm_theta = samples.get('normalised_theta')
        ys = np.array(norm_theta)

        # Ensure that the ys are in the same order as fp.raw_members
        #
        # In older versions of the code, column names were only saved with the
        # 'theta' dataset (not norm_theta). This is why we still refer to this
        # as fallback.
        try:
            colnames = list(norm_theta.attrs['columns'])
        except:
            theta = samples.get('theta')
            colnames = list(theta.attrs['columns'])
        permlist = [colnames.index(cn) for cn in column_order]
        ys = ys[:, permlist]
        return xs, ys

    def get_xs(self) -> Any:
        """Just return all the xs (photometric measurements) in the dataset

        Returns type Any (not Tensor, or np.ndarray) because the
        transformations could be arbitrary.
        """
        xs = self.dataset[:, :self._x_dim]
        for tr in self.transforms:
            xs = tr(xs)
        return xs.squeeze()

    def get_ys(self) -> Any:
        """Return all the y values (physical parameters) in the dataset

        Returns type Any (not Tensor, or np.ndarray) because the
        transformations could be arbitrary.
        """
        ys = self.dataset[:, self._x_dim:]
        for tr in self.transforms:
            ys = tr(ys)
        return ys.squeeze()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: Union[int, list[int], Tensor]) -> tuple[
            tensor_like, tensor_like]:

        if isinstance(idx, Tensor):
            idx = idx.to(dtype=t.int).tolist()

        data = self.dataset[idx]
        if data.ndim == 1:
            data = np.expand_dims(data, 0)
        xs, ys = data[:, :self._x_dim], data[:, self._x_dim:]

        for tr in self.transforms:
            xs, ys = tr(xs), tr(ys)

        # both np.ndarray and torch.Tensor implement `squeeze`
        return (xs.squeeze(), ys.squeeze())


# For backwards compatability
GalaxyDataset = InMemoryGalaxyDataset


def load_simulated_data(
        path: str, split_ratio: float = 0.8, batch_size: int = 32,
        test_batch_size: Optional[int] = None,
        normalise_phot: Optional[Callable[[Any], Any]] = None,
        transforms: list[Callable[[Any], Any]] = [t.from_numpy],
        x_transforms: list[Callable[[Any], Any]] = [],
        y_transforms: list[Callable[[Any], Any]] = [],
        split_seed: int = 0
        ) -> tuple[DataLoader, DataLoader]:
    """Load simulated (theta, photometry) data as train and test data loaders.

    Args:
        path: file path to the .hdf5 file containing simulated data
        split_ratio: train / test split ratio
        batch_size: training batch size (default 32)
        test_batch_size: optional different batch size for testing (defaults to
            `batch_size`)
        normalise_phot: an optional normalisation transformation to apply to
            simulated photometry.
        transforms: list of transformations to apply to data before returning
        x_transforms: any photometry specific transformations
        y_transforms: any parameter specific transformations
        split_seed: PyTorch Generator Seed for reproducible train/test splits.

    Returns:
        tuple[DataLoader, DataLoader]: train and test DataLoaders, respectively
    """
    tbatch_size = test_batch_size if test_batch_size is not None else batch_size

    cuda_kwargs = {'num_workers': 8, 'pin_memory': True}
    train_kwargs: dict[str, Any] = {
        'batch_size': batch_size, 'shuffle': True} | cuda_kwargs
    test_kwargs: dict[str, Any] = {
        'batch_size': tbatch_size, 'shuffle': True} | cuda_kwargs

    dataset = InMemoryGalaxyDataset(path, normalise_phot, transforms,
                                    x_transforms, y_transforms)

    n_train = int(len(dataset) * split_ratio)
    n_test = len(dataset) - n_train

    rng = t.Generator().manual_seed(split_seed)
    train_set, test_set = random_split(dataset, [n_train, n_test], rng)

    train_loader = DataLoader(train_set, **train_kwargs)
    test_loader = DataLoader(test_set, **test_kwargs)

    return train_loader, test_loader


class Squareplus(nn.Module):
    def __init__(self, a=2):
        super().__init__()
        self.a = a
    def forward(self, x):
        """The 'squareplus' activation function: has very similar properties to
        softplus, but is computationally cheaper and more configurable.
            - squareplus(0) = 1 (softplus(0) = ln 2)
            - gradient diminishes more slowly for negative inputs.
            - ReLU = (x + sqrt(x^2))/2
            - 'squareplus' becomes smoother with higher 'a'
        """
        return (x + t.sqrt(t.square(x)+self.a*self.a))/2


def squareplus_f(x, a=2):
    """Functional version of the 'squareplus' activation function. See
    `Squareplus` for more information.
    """
    return (x + t.sqrt(t.square(x)+a*a))/2


def centre_phot_np(x: np.ndarray) -> np.ndarray:
    """Normalise a numpy array along 0th dimension in log-space.

    WARNING: using this without saving the mean and std for identical
    application of normalisation to test data is not recommended.

    Args:
        x: array to normalise; assumed to be a tensor of photometric
            observations.

    Returns:
        np.ndarray: the normalised photometry.
    """
    x_log = np.log(x)
    return (x_log - x_log.mean(0)) / x_log.std(0)


def normalise_phot_np(x: np.ndarray) -> np.ndarray:
    """Normalise the photometry, such that that log-observations across all
    filters sum to 1.

    Args:
        x: raw photometric observations

    Returns:
        np.ndarray: normalised photometric observations
    """
    x_log = np.log(x)
    # get the total magnitude of the log-photometry observations
    mags = x_log.sum(1)
    # ensure that the observation magnitudes across all filters sum to 1
    return x_log / mags[:,None]


def maggies_to_colours(x: Tensor) -> Tensor:
    """Compute the 'colours' from maggies [f₁, f₂, …, fₙ]:

        {(fᵢ - fⱼ) | i, j ∈ {1, …, N}, i < j}

    Args:
        x: Matrix of N input points [N, D]; D filters per row

    Returns:
        Tensor: an [N, C] array, for C = N(N-1)/2, the number of colours.
    """
    i = t.triu_indices(x.shape[-1]-1, x.shape[-1]-1)+1
    mat = x[...,None].expand(*x.shape, x.shape[-1])
    return mat[...,i[0],i[1]] - mat[...,i[1],i[0]]


def maggies_to_colours_np(x_np: np.ndarray) -> np.ndarray:
    """Numpy version of the above."""
    return maggies_to_colours(t.from_numpy(x_np)).numpy()


def get_colours_length(filters: int) -> int:
    """Returns the number of colours for the specified number of filters.
    """
    return int(filters * (filters - 1) / 2)


# Testing Utilities -----------------------------------------------------------
# The following functions are for use in the accompanying unit tests and
# notebooks.


def _load_mnist(batch_size: int = 64, dtype: t.dtype = t.float64,
               device: t.device = t.device('cpu')
               ) -> tuple[DataLoader, DataLoader]:
    """(Down)load MNIST dataset in ./data/testdata, and return training and
    test DataLoaders using specified batch_size.
    """
    warnings.filterwarnings('ignore', category=UserWarning)  # see torchvision pr #4184

    transform = transforms.Compose([
        lambda x: np.array(x),
        transforms.ToTensor(),
        lambda x: x.to(device, dtype)
    ])

    train_set = datasets.MNIST('./data/testdata', train=True, download=True,
                               transform=transform)
    test_set = datasets.MNIST('./data/testdata', train=False, download=True,
                              transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def _onehot(idx: Tensor, n: int) -> Tensor:
    """Turns an index into a one-hot encoded vector, of length n"""
    assert t.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = t.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot
