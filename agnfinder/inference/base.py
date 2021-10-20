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
"""Abstract base class for conditional VAE used for parameter inference from
photometry.
"""

import os
import abc
import logging
import torch as t
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import Union, Optional, Type, Callable

from agnfinder.types import Tensor, DistParams, arch_t


class MLP(nn.Module):
    """A base neural network class for simple feed-forward architectures."""

    def __init__(self, arch: arch_t, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        """Initialises a neural network based on the description in `arch`.

        Args:
            arch: neural network architecture description
            device: device memory on which to store parameters
            dtype: datatype to use for parameters
        """
        super().__init__()
        self.is_module: bool = True
        self._arch = arch  # keep a record of arch description
        layer_sizes = arch.layer_sizes

        self.MLP = nn.Sequential()
        for i, (j, k) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name=f'L{i}', module=nn.Linear(j, k))
            if arch.batch_norm:
                # TODO remove bias from previous layer if BN
                self.MLP.add_module(name=f'BN{i}', module=nn.BatchNorm1d(k))
            self.MLP.add_module(name=f'A{i}', module=arch.activations[i])
        self.MLP.to(device=device, dtype=dtype)

        h_n = layer_sizes[-1]
        self.heads: nn.ModuleList = nn.ModuleList()
        for i, h in enumerate(arch.head_sizes):
            this_head = nn.Sequential()
            this_head.add_module(name=f'H{i}', module=nn.Linear(h_n, h))
            if arch.head_activations[i] is not None:
                this_head.add_module(name=f'HA{i}',
                                     module=arch.head_activations[i])
            this_head.to(device=device, dtype=dtype)
            self.heads.append(this_head)

    def forward(self, x: Tensor) -> Union[Tensor, DistParams]:
        """Forward pass through the network

        Args:
            x: input tensor

        Returns:
            Union[Tensor, DistParams]: Return single tensor if there is a single
                head, otherwise a list of tensors (DistParams) with each element
                the output of a head.
        """
        y = self.MLP(x)
        heads = []
        heads = [h(y) for h in self.heads]
        if len(heads) == 1:
            return heads[0]
        return heads

    @property
    def out_len(self) -> int:
        """Number of tensor parameters returned"""
        return len(self._arch.head_sizes)

    def __repr__(self) -> str:
        """Prints representation of NN architecture"""
        a = f'{type(self).__name__} Network Architecture'
        b = int((78 - len(a))/2)

        r = f'\n\n{b*"~"} {a} {b*"~"}\n{self.MLP}'
        for i, h in enumerate(self.heads):
            r += f'\nHead {i}:\n{h}'
        r += f'\n{79*"~"}\n'
        return r


class _CVAE_Dist(object):
    """Base distribution class for use with encoder, decoder and prior

    Note: currently the dtype/device is the same as is being used for the rest
        of training. However, if the PRNG is only available on a different
        device, this may make sampling slow (e.g. GPU training, where PRNG is
        on CPU will cause lots of data transfers). TODO: implement more
        granular device/dtype control.
    """

    def __init__(self, batch_shape: t.Size =t.Size(),
                 event_shape: t.Size =t.Size(),
                 device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self.device: t.device = device
        self.dtype: t.dtype = dtype

    def __call__(self, *args, **kwargs):
        self.log_prob(*args, **kwargs)

    def _extended_shape(self, sample_shape=t.Size()) -> t.Size:
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        Args:
            sample_shape: the size of the sample to be drawn.
        """
        if not isinstance(sample_shape, t.Size):
            sample_shape = t.Size(sample_shape)
        return t.Size(sample_shape + self._batch_shape + self._event_shape)

    @abc.abstractmethod
    def log_prob(self, value: Tensor, nojoint: bool = False) -> Tensor:
        """
        Returns the log of the probability density/mass function evaluated at
        value.
        Set nojoint to True to avoid finding the (iid assumed) joint
        probability along the 1st dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        """
        Generates a `sample_shape` shaped sample or `sample_shape` shaped batch
        of samples if the distribution parameters are batched.
        """
        raise NotImplementedError


class _CVAE_RDist(_CVAE_Dist):
    """Explicit Rdist class for distributions implementing reparametrised
    sampling.
    """

    def sample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        return self.rsample(sample_shape).to(self.device, self.dtype)

    @abc.abstractmethod
    def rsample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
        raise NotImplementedError


class CVAEPrior(MLP, abc.ABC):
    """Base prior class for CVAEs

    This implements the (optional) prior network which parametrises
    p_{theta}(z | x)
    """
    def __init__(self, arch: Optional[arch_t],
                 latent_dim: int,
                 device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        abc.ABC.__init__(self)
        if arch is not None:
            MLP.__init__(self, arch, device, dtype)
        else:
            self.is_module: bool = False
            self._parameters = {}
            self._modules = {}
            self._buffers = {}
            self._state_dict_hooks = {}
        self.latent_dim = latent_dim
        self.device, self.dtype = device, dtype

    def __call__(self, x: Tensor) -> _CVAE_Dist:
        if self.is_module:
            return self.get_dist(self.forward(x))
        return self.get_dist(None)

    def __repr__(self) -> str:
        return type(self).__name__

    @abc.abstractmethod
    def get_dist(self, dist_params: Optional[Union[Tensor, DistParams]] = None
                 ) -> _CVAE_Dist:
        raise NotImplementedError


class CVAEEnc(MLP, abc.ABC):
    """Base encoder class for CVAEs

    This implements the recognition network which parametrises
    q_{phi}(z | y, x)
    """

    def __init__(self, arch: arch_t, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        abc.ABC.__init__(self)
        MLP.__init__(self, arch, device, dtype)
        self.device, self.dtype = device, dtype

    def __call__(self, y: Tensor, x: Tensor) -> _CVAE_RDist:
        return self.get_dist(
                    self.forward(
                        t.cat((y, x), -1)))

    def __repr__(self) -> str:
        return type(self).__name__

    @abc.abstractmethod
    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_RDist:
        raise NotImplementedError


class CVAEDec(MLP, abc.ABC):
    """Base decoder class for CVAEs

    This implements the generator network which parametrises
    p_{theta}(y | z, x)
    """

    def __init__(self, arch: arch_t, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        abc.ABC.__init__(self)
        MLP.__init__(self, arch, device, dtype)
        self.device, self.dtype = device, dtype

    def __call__(self, z: Tensor, x: Tensor) -> _CVAE_Dist:
        return self.get_dist(
                    self.forward(
                        t.cat((z, x), -1)))

    def __repr__(self) -> str:
        return type(self).__name__

    @abc.abstractmethod
    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_Dist:
        raise NotImplementedError


# CVAE Description ------------------------------------------------------------
# This unfortunately cannot go in agnfinder.types for this causes a circular
# dependency.

class CVAEParams(abc.ABC):
    """Configuration class for CVAE.

    This defines some properties which must be provided, and additionally
    performs some validation on those user-provided values.
    """
    def __init__(self):
        super().__init__()
        ri = self.enc_arch.in_shape
        if ri != self.data_dim + self.cond_dim:
            raise ValueError((
                f'Input dimensions of encoder network ({ri}) '
                f'must equal data_dim ({self.data_dim}) + '
                f'cond_dim ({self.cond_dim}).'))

        if self.prior_arch is not None:
            pi = self.prior_arch.in_shape
            if pi != self.cond_dim:
                raise ValueError((
                    f'Input dimensions of prior network ({pi}) '
                    f'must equal cond_dim ({self.cond_dim})'))

        gi = self.dec_arch.in_shape
        if gi != self.latent_dim + self.cond_dim:
            raise ValueError((
                f'Input dimensions of decoder network ({gi}) '
                f'must euqal latent_dim ({self.latent_dim}) + '
                f'cond_dim ({self.cond_dim})'))

    @property
    def cond_dim(self) -> int:
        """Length of 1D conditioning information vector"""
        raise NotImplementedError

    @property
    def data_dim(self) -> int:
        """Length of the perhaps (flattened) 1D data vector, y"""
        raise NotImplementedError

    @property
    def latent_dim(self) -> int:
        """Length of the latent vector, z"""
        raise NotImplementedError

    @property
    def adam_lr(self) -> float:
        """Learning rate to use with Adam optimizer"""
        return 1e-3

    @property
    def prior(self) -> Type[CVAEPrior]:
        """Reference to the prior class to use."""
        raise NotImplementedError

    @property
    def prior_arch(self) -> Optional[arch_t]:
        """Architecture of 'prior network' p_{theta_z}(z | x)"""
        return None

    @property
    def encoder(self) -> Type[CVAEEnc]:
        """Reference to the encoder / recognition class to use"""
        raise NotImplementedError

    @property
    def enc_arch(self) -> arch_t:
        """Architecture of 'recognition network' q_{phi}(z | y, x)"""
        raise NotImplementedError

    @property
    def decoder(self) -> Type[CVAEDec]:
        """Reference to the decoder / generation class to use"""
        raise NotImplementedError

    @property
    def dec_arch(self) -> arch_t:
        """Architecture of 'generator network' p_{theta_y}(y | z, x)"""
        raise NotImplementedError


# Main CVAE Base Class ========================================================


class CVAE(nn.Module, abc.ABC):
    """The main base Conditional VAE class

    You must provide the following distributions in the configuration.
    - q_{phi}(z | y, x)   approximate posterior / encoder
    - p_{theta}(z | x)    prior / encoder
    - p_{theta}(y | z, x) generator / decoder

    You can optionally override the `ELBO` method (for instance to implement KL
    warmup). You can also override the `trainmodel` method to implement a
    non-standard training procedure, as well as the `preprocess` method for
    custom data pre-processing.
    """

    def __init__(self, cp: CVAEParams,
                 device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64,
                 logging_callbacks: list[Callable] = [],
                 overwrite_results: bool = False) -> None:
        """Initialise a CVAE

        Args:
            cp: CVAE parameters (usually from config.py)
            device: device on which to store the model
            dtype: data type to use in Tensors
            logging_callbacks: list of callables accepting this CVAE instance
            overwrite_results: whether to replace previous model when saving
        """
        super().__init__()  # init nn.Module, ABC
        self.device = device
        self.dtype = dtype
        self.cp = cp
        self.latent_dim = cp.latent_dim
        self.logging_callbacks = logging_callbacks

        self.prior = cp.prior(cp.prior_arch, cp.latent_dim, device, dtype)
        self.encoder = cp.encoder(cp.enc_arch, device, dtype)
        self.decoder = cp.decoder(cp.dec_arch, device, dtype)

        self.is_trained: bool = False
        self.overwrite_results = overwrite_results
        self.savepath_cached: Optional[str] = None

        # self.opt = t.optim.Adam(self.parameters(), lr=cp.adam_lr)

        if self.prior.is_module:
            self.prior_opt = t.optim.Adam(self.prior.parameters(), lr=cp.adam_lr)
        self.enc_opt = t.optim.Adam(self.encoder.parameters(), lr=cp.adam_lr)
        self.dec_opt = t.optim.Adam(self.decoder.parameters(), lr=cp.adam_lr)

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

    def ELBO(self, logpy: Tensor, logpz: Tensor, logqz: Tensor, i: int, tot: int
             ) -> Tensor:
        """Compute and return the ELBO.

        You could override this method to, for instance, anneal the temperature
        of the KL term during training.

        Args:
            logpy: log-likelihood term; log p_{theta}(y | z, x)
            logpz: log prior term; log p_{theta}(z | x)
            logqz: log approx posterior term; log q_{phi}(z | y, x)
            i: current iteration
            t: total number of iterstions in training process

        Returns:
            Tensor: the batch of single-datapoint ELBOs
        """
        if logpy.isnan().any():
            logging.warn('logpy is NaN')
            logpy = logpy.nan_to_num()
        if logpz.isnan().any():
            logging.warn('logpz is NaN')
            logpz = logpz.nan_to_num()
        if logqz.isnan().any():
            logging.warn('logqz is NaN')
            logqz = logqz.nan_to_num()
        return logpy + logpz - logqz

    def trainmodel(self, train_loader: DataLoader, epochs: int = 10,
                   log_every: int = 100) -> None:
        """Trains the CVAE

        Args:
            train_loader: DataLoader for training data
            epochs: number of epochs to train for
            log_every: logging frequency (iterations, not epochs)
        """
        self.train()
        b = train_loader.batch_size
        assert isinstance(b, int)
        ipe = len(train_loader) * b  # 'iterations per epoch'
        t = epochs * ipe

        for e in range(epochs):
            for i, (x, y) in enumerate(train_loader):

                # x is photometry, y are parameters (theta)
                x, y = self.preprocess(x, y)

                # Get q_{phi}(z | y, x)
                q: _CVAE_RDist = self.encoder(y, x)
                z = q.rsample()

                # Get p_{theta}(z | x)
                pr: _CVAE_Dist = self.prior(x)

                # Get p_{theta}(y | z, x)
                p: _CVAE_Dist = self.decoder(z, x)

                logpy = p.log_prob(y)
                logpz = pr.log_prob(z)
                logqz = q.log_prob(z)

                ELBO = self.ELBO(logpy, logpz, logqz, (e*ipe) + (i*b), t)

                loss = -(ELBO.mean(0))
                if self.prior.is_module:
                    self.prior_opt.zero_grad()
                self.enc_opt.zero_grad()
                self.dec_opt.zero_grad()

                loss.backward()

                if self.prior.is_module:
                    self.prior_opt.step()
                self.enc_opt.step()
                self.dec_opt.step()

                if i % log_every == 0 or i == len(train_loader)-1:
                    # Run through all logging functions
                    [cb(self) for cb in self.logging_callbacks]
                    logging.info(
                        "Epoch: {:02d}/{:02d}, Batch: {:03d}/{:d}, Loss {:9.4f}"
                        .format(e, epochs, i, len(train_loader)-1, loss.item()))
        self.is_trained = True

    def log_cond(self, y: Tensor, x: Tensor, K: int = 1000) -> Tensor:
        """Evaluates log p_{theta}(y | x)

        This uses an Monte-Carlo approximation to the marginal likelihood,
        using K samples.

        Args:
            y: the parameter values to find the (conditional) likelihood of
            x: the conditioning photometry values
            K: the number of MC sapmles to use

        Returns:
            Tensor: p_{theta}(y | x) for the provided ys.
        """
        if not self.is_trained:
            logging.warn("CVAE is not yet trained!")

        if y.shape != x.shape:
            try:
                # attempt to broadcase x to be the same shape as y
                x = x.expand((y.size(0), -1))
                assert y.shape == x.shape
            except:
                raise RuntimeError((
                    f'Cannot call log_cond with y of shape: {y.shape} '
                    f'and x of shape: {x.shape}'
                ))

        self.eval()
        x, y = self.preprocess(x, y)

        with t.inference_mode():
            q: _CVAE_RDist = self.encoder(y, x)
            z = q.rsample(t.Size((K,)))
            logqz = q.log_prob(z, nojoint=True).sum(-1)

            pr: _CVAE_Dist = self.prior(x)
            logpz = pr.log_prob(z, nojoint=True).sum(-1)

            flat_zs = z.flatten(0,1)
            tmp_xs = x.repeat_interleave(K, dim=0)
            tmp_ys = y.repeat_interleave(K, dim=0)
            p: _CVAE_Dist = self.decoder(flat_zs, tmp_xs)
            logpy = p.log_prob(tmp_ys).reshape((K, -1))

            return (logpy + logpz -logqz).mean(0)

    def test_generator(self, test_loader: DataLoader) -> float:
        """Test the generator network, p_{theta}(y | z, x)

        Args:
            test_loader: data loader for test set

        Returns:
            float: test loss
        """
        self.eval()
        loss = t.zeros(1, device=self.device, dtype=self.dtype)
        for x, y in test_loader:
            x, y = self.preprocess(x, y)
            pr: _CVAE_Dist = self.prior(x)
            z = pr.sample(t.Size((x.size(0),)))
            p: _CVAE_Dist = self.decoder(z, x)
            NLL = p.log_prob(y)
            loss -= NLL.mean(0)
        return float(loss / len(test_loader))


    def fpath(self) -> str:
        """Returns a file path to save the model to, based on parameters."""
        # TODO generate a more unique model name based on network architectures
        # (not sequence)
        if self.savepath_cached is None:
            base = './results/models/'
            name = f'{self.prior}_{self.encoder}_{self.decoder}'
            if self.overwrite_results:
                self.savepath_cached = f'{base}{name}.pt'
            else:
                n = 0
                while os.path.exists(f'{base}{name}_{n}.pt'):
                    n+=1
                self.savepath_cached = f'{base}{name}_{n}.pt'
        return self.savepath_cached


# For use in configuration file.
cvae_t = Type[CVAE]
