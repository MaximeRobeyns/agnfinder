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
"""Entrypoint for inference tasks."""

import os
import re
import logging
import torch as t
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Any, Callable, Type
from torchvision import transforms
from torch.utils.data import DataLoader

from agnfinder import config as cfg
from agnfinder.types import Tensor, tensor_like
from agnfinder.utils import ConfigClass


# Abstract inference parameters ------------------------------------------------


class InferenceParams(ConfigClass):
    """Abstract configuration for the inference section of the progra."""

    @property
    @abstractmethod
    # Type is `Any` due to poorly thought out design causing a circular
    # dependency. The refactor is probably not worth the headache.
    def model(self) -> Any:
        """The model to use"""
        pass

    @property
    def split_ratio(self) -> float:
        """Train / test split ratio"""
        return 0.9

    @property
    def logging_frequency(self) -> int:
        """Number of iterations between logs"""
        return 1000

    @property
    def dataset_loc(self) -> str:
        """Filepath to the hdf5 file or directory of hdf5 files to use"""
        return ''

    @property
    def retrain_model(self) -> bool:
        """Whether to retrain an identical (existing) model"""
        return False

    @property
    def use_existing_checkpoints(self) -> bool:
        """Whether to attempt to use checkpoints (if any). If set to False, any
        previous checkpoints (for identical model fpath) will be deleted to
        allow new checkpoints to be saved."""
        return True

    @property
    def overwrite_results(self) -> bool:
        """If `retrain_model`, should we overwrite existing result on disk?"""
        return False

    @property
    def ident(self) -> str:
        """An identifier for the training run. This could contain the name of
        the dataset, or some other identifying feature of this model.

        This is appended to `fpath` to allow for saving/loading specific model
        versions.
        """
        return ""


# Abstract model ---------------------------------------------------------------


class ModelParams(ConfigClass, ABC):

    @property
    @abstractmethod
    def epochs(self) -> int:
        """The number of epochs to train the model for.

        Putting this parameter here risks incurring a 'type error'; this is
        really an inference parameter (how long we train the model for),
        however since this has such a large effect on the resulting saved
        model, we prefer to associate it with the model itself.
        """
        pass

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """The mini-batch size"""
        pass

    @property
    @abstractmethod
    def dtype(self) -> t.dtype:
        """The data type to use with this model. e.g. torch.float32"""
        pass

    @property
    def device(self) -> t.device:
        """The device on which to run this model."""
        return t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    @property
    @abstractmethod
    def cond_dim(self) -> int:
        """Length of 1D conditioning information vector"""
        pass

    @property
    @abstractmethod
    def data_dim(self) -> int:
        """Length of the perhaps (flattened) 1D data vector, y"""
        pass


class Model(nn.Module, ABC):
    """Base model class for AGNFinder."""

    def __init__(self, mp: ModelParams, overwrite_results: bool = False,
                 logging_callbacks: list[Callable] = []):
        """Initialises a model taking the configuration parameters

        Args:
            mp: model parameters
            overwrite_results: whether to overwrite a model with the same
                filepath (read, same parameters) at the ned of training. Default:
                True
            logging_callbacks: list of callables accepting this model instance;
                often used for visualisations and debugging.
        """
        super().__init__()

        # For convenience, make some attributes of mp also attributes of Model:
        self.dtype = mp.dtype
        self.device = mp.device
        self.data_dim = mp.data_dim
        self.cond_dim = mp.cond_dim
        self.batch_size = mp.batch_size
        self.epochs = mp.epochs

        self.is_trained: bool = False
        self.overwrite_results = overwrite_results
        self.savepath_cached: str = ""
        self.logging_callbacks = logging_callbacks

        if self.device == t.device('cuda'):
            self.to(self.device, self.dtype)

    def __init_subclass__(cls):
        # Apply 'decorators' to certain methods for inheriting classes.
        # This is not a particularly pretty pattern to inherit decorators, but
        # it works well enough...
        try:
            # only apply decorator to the last class in the inheritance hierarchy:
            assert cls._sub_init
        except AttributeError:
            cls.__repr__ = Model._wrap_lines(cls.__repr__)
            cls.trainmodel = Model._save_results(cls.trainmodel)
            cls._sub_init = True
        return cls

    @staticmethod
    def _wrap_lines(__repr__: Callable[[Any], str]) -> Callable[[Any], str]:
        """Wraps model description after every 80 characters for more
        ~aesthetic~ plotting / logging.
        """
        def _f(self) -> str:
            string = __repr__(self)
            words = string.split(' ')
            length: int = 0
            for i in range(len(words)):
                length += len(words[i])
                if length >= 80:
                    words[i-1] = words[i-1] + '\n'  # i-1 should be fine...
                    length = len(words[i])
            return " ".join(words).replace("\n ", "\n")

        return _f

    @staticmethod
    def _save_results(trainmodel: Callable[..., None]) -> Callable[..., None]:
        """Decorator for the training method `trainmodel` which caches trained
        models on disk avoiding unnecessary re-training of identical models,
        and preventing users from forgetting to save models at the end of
        training!

        Args:
            trainmodel: training function from inheriting class
        """
        def _f(self, loader: DataLoader, ip: InferenceParams, *args,
               **kwargs) -> None:
            # Attempt to load the model from disk instead of re-training an
            # identical model.
            savepath: str = self.fpath(ip.ident)
            if not ip.retrain_model:
                try:
                    logging.info(
                        f'Attempting to load {self.name} model from {savepath}')
                    self.load_state_dict(t.load(savepath))
                    self.is_trained = True
                    logging.info(f'Successfully loaded')
                    # unsure if this is necessary, but for good measure...
                    self.to(self.device, self.dtype)
                    return
                except:
                    logging.info(
                        f'Could not load model at {savepath}. Training...')

            # Do the training
            trainmodel(self, loader, ip, *args, **kwargs)
            self.is_trained = True
            logging.info(f'Trained {self}.')

            # Save the model to disk
            t.save(self.state_dict(), savepath)
            logging.info(
                f'Saved {self.name} model as: {savepath}')
        return _f


    def checkpoint(self, ident: str='') -> None:

        # remove extension from file
        checkpoint_dir = '.'.join(self.fpath(ident).split('.')[:-1])
        assert checkpoint_dir != '', "fpath must contain an extension; `.pt` recommended"

        # check that directory exists
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        checkpoints = os.listdir(checkpoint_dir)
        r = re.compile('checkpoint_(?P<Num>\d+)\.pt$')
        ints = []
        for c in checkpoints:
            match = r.match(c)
            if match is not None and match.group('Num') is not None:
                ints.append(int(match.group('Num')))

        n = 1 if len(ints) == 0 else max(ints) + 1
        savepath = checkpoint_dir + f'/checkpoint_{n}.pt'

        # Save the checkpoint to disk
        t.save(self.state_dict(), savepath)
        logging.info(
            f'Saved {self.name} checkpoint to {savepath}.')


    def attempt_checkpoint_recovery(self, ip: InferenceParams) -> int:
        """Attempts to load the latest checkpoint file.

        Returns:
            int: the checkpoint number
        """

        checkpoint_dir = '.'.join(self.fpath(ip.ident).split('.')[:-1])
        if checkpoint_dir == '':
            logging.warn(('Will not be able to save a checkpoint with fpath '
                          'of {self.fpath(ip.ident)}!'))
            return 0

        if not os.path.isdir(checkpoint_dir) or not ip.use_existing_checkpoints:
            logging.info(f'No previous checkpoints found at {checkpoint_dir}')
            return 0

        checkpoints = os.listdir(checkpoint_dir)
        if not ip.use_existing_checkpoints:
            logging.info(f'Removing old checkpoints from {checkpoint_dir}')
            r = re.compile('checkpoint_\d+\.pt$')
            files = [c for c in checkpoints if r.match(c)]
            for c in files:
                os.remove(os.path.join(checkpoint_dir, c))
            return 0
        else:
            ints = []
            r = re.compile('checkpoint_(?P<Num>\d+)\.pt$')
            for c in checkpoints:
                match = r.match(c)
                if match is not None and match.group('Num') is not None:
                    ints.append(int(match.group('Num')))

        if len(ints) == 0:
            logging.info(f'No previous checkpoints found at {checkpoint_dir}')
            return 0

        latest_n = max(ints)
        while latest_n > 0:
            try:
                logging.info(f'Attempting to load checkpoint {latest_n}')
                cpath = os.path.join(checkpoint_dir, f'checkpoint_{latest_n}.pt')
                self.load_state_dict(t.load(cpath))
                self.to(self.device, self.dtype)
                return latest_n
            except:
                logging.warning(f'Could not load checkpoint {latest_n}')
                latest_n -= 1
        logging.warning(f'Checkpoint loading failed')
        return 0


    @abstractmethod
    def __repr__(self) -> str:
        """Classes inheriting `Model` _should_ override this method to give a
        more descriptive representation of the model."""
        return (f'{self.name} trained for {self.epochs} epochs with '
                f'batch size {self.batch_size}')

    @property
    @abstractmethod
    def name(self) -> str:
        """The name with which to refer to this model (e.g. for saving to disk)
        """
        pass

    @abstractmethod
    def fpath(self, ident: str='') -> str:
        """Returns a file path to save the model to, based on its parameters.

        An optional identifier, if provided, is appended to the filename before
        the extension.
        """
        raise NotImplementedError

    def preprocess(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Perform any necessary pre-processing to the data before training.

        If overriding this method, always remember to cast the data to
        self.dtype and put it on self.device.

        Args:
            x: the input (e.g. predictor variables)
            y: the output (e.g. response variables)

        Returns:
            tuple[Tensor, Tensor]: the transformed data.
        """
        return x.to(self.device, self.dtype), y.to(self.device, self.dtype)

    @abstractmethod
    def trainmodel(self, train_loader: DataLoader, ip: InferenceParams,
                   *args, **kwargs) -> None:
        """Train the model.

        Args:
            train_loader: DataLoader to load the training data.
            ip: the inference parameters describing the training procedure.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, x: tensor_like, n_samples: int = 1000, *args, **kwargs) -> Tensor:
        """A convenience method for drawing (conditional) samples from p(y | x)
        for a single conditioning point.

        Args:
            cond_data: the conditioning data; x
            n_samples: the number of samples to draw

        Returns:
            Tensor: a tensor of shape [n_samples, data_dim]
        """
        raise NotImplementedError


model_t = Type[Model]


# Inference ====================================================================


if __name__ == '__main__':
    """
    Load configurations, train the selected model and save to disk (or just
    load from disk, as appropriate), then draw samples for a galaxy, outputting
    the plot in the results directory.
    """
    from agnfinder.inference import utils

    cfg.configure_logging()

    ip = cfg.InferenceParams()  # inference procedure parameters

    # this is poor form :(
    from agnfinder.inference import CMADE, CVAE
    mp: ModelParams = cfg.SANParams()
    if ip.model == CMADE:
        mp = cfg.MADEParams()
    elif ip.model == CVAE:
        mp = cfg.CVAEParams()

    train_loader, test_loader = utils.load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=mp.batch_size,
        normalise_phot=utils.normalise_phot_np,
        transforms=[
            transforms.ToTensor()
        ],
        x_transforms=[utils.maggies_to_colours_np])
    logging.info('Created data loaders')

    model = ip.model(mp)
    logging.info('Initialised {model.name} model')

    # NOTE: uses cached model (if available), and saves to disk after training.
    model.trainmodel(train_loader, ip)
    logging.info('Trained {model.name} model')

    # TODO sample and save result with some descriptive filename.
    # x, _ = nbu.new_sample(test_loader)
    # model.sample(x, n_samples=1000)
    # logging.info('Successfully sampled from model')
