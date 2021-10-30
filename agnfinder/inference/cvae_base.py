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

import torch as t
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Union, Optional

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

    @abstractmethod
    def log_prob(self, value: Tensor, nojoint: bool = False) -> Tensor:
        """
        Returns the log of the probability density/mass function evaluated at
        value.
        Set nojoint to True to avoid finding the (iid assumed) joint
        probability along the 1st dimension.
        """
        raise NotImplementedError

    @abstractmethod
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

    @abstractmethod
    def rsample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
        raise NotImplementedError


class CVAEPrior(MLP, ABC):
    """Base prior class for CVAEs

    This implements the (optional) prior network which parametrises
    p_{theta}(z | x)
    """
    def __init__(self, arch: Optional[arch_t],
                 latent_dim: int,
                 device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        ABC.__init__(self)
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

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the prior"""
        raise NotImplementedError

    @abstractmethod
    def get_dist(self, dist_params: Optional[Union[Tensor, DistParams]] = None
                 ) -> _CVAE_Dist:
        raise NotImplementedError


class CVAEEnc(MLP, ABC):
    """Base encoder class for CVAEs

    This implements the recognition network which parametrises
    q_{phi}(z | y, x)
    """

    def __init__(self, arch: arch_t, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        ABC.__init__(self)
        MLP.__init__(self, arch, device, dtype)
        self.device, self.dtype = device, dtype

    def __call__(self, y: Tensor, x: Tensor) -> _CVAE_RDist:
        return self.get_dist(
                    self.forward(
                        t.cat((y, x), -1)))

    def __repr__(self) -> str:
        return type(self).__name__

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the encoder"""
        raise NotImplementedError

    @abstractmethod
    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_RDist:
        raise NotImplementedError


class CVAEDec(MLP, ABC):
    """Base decoder class for CVAEs

    This implements the generator network which parametrises
    p_{theta}(y | z, x)
    """

    def __init__(self, arch: arch_t, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        ABC.__init__(self)
        MLP.__init__(self, arch, device, dtype)
        self.device, self.dtype = device, dtype

    def __call__(self, z: Tensor, x: Tensor) -> _CVAE_Dist:
        return self.get_dist(
                    self.forward(
                        t.cat((z, x), -1)))

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the decoder"""
        raise NotImplementedError

    def __repr__(self) -> str:
        return type(self).__name__

    @abstractmethod
    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_Dist:
        raise NotImplementedError
