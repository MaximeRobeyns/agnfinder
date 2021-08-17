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

import abc
import torch as t
import torch.nn as nn

from typing import Union

from agnfinder.types import Tensor, Distribution, DistParam, arch_t


class MLP(nn.Module):

    def __init__(self, arch: arch_t, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        """Base neural network class for simple feed-forward architectures.

        Args:
            arch: neural network architecture description
            device: device memory on which to store parameters
            dtype: datatype to use for parameters
        """
        super().__init__()
        self._arch = arch  # keep a record of arch description
        layer_sizes = arch.layer_sizes

        self.MLP = nn.Sequential()
        for i, (j, k) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name=f'L{i}', module=nn.Linear(j, k))
            if arch.batch_norm:
                self.MLP.add_module(name=f'BN{i}', module=nn.BatchNorm1d(k))
            self.MLP.add_module(name=f'A{i}', module=arch.activations[i])
        self.MLP.to(device=device, dtype=dtype)

        h_n = layer_sizes[-1]
        self.heads: list[nn.Module] = []
        for i, h in enumerate(arch.head_sizes):
            this_head = nn.Sequential()
            this_head.add_module(name=f'H{i}', module=nn.Linear(h_n, h))
            if arch.head_activations[i] is not None:
                this_head.add_module(name=f'HA{i}',
                                     module=arch.head_activations[i])
            this_head.to(device=device, dtype=dtype)
            self.heads.append(this_head)

    def forward(self, x: Tensor) -> Union[Tensor, DistParam]:
        """Forward pass through the network

        Args:
            x: input tensor

        Returns:
            Union[Tensor, DistParam]: Return single tensor if there is a single
                head, otherwise a list of tensors (DistParam) with each element
                the output of a head.
        """
        y = self.MLP(x)
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


class RecognitionNet(MLP, abc.ABC):
    """An abstract recognition network; implementing q_{phi}(z | y, x)."""

    def __init__(self, arch: arch_t, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        MLP.__init__(self, arch, device, dtype)
        abc.ABC.__init__(self)

    @abc.abstractmethod
    def distribution(self, y: Tensor, x: Tensor) -> Distribution:
        raise NotImplementedError


class PriorNet(MLP, abc.ABC):
    """Abstract 'prior' network; implementing p_{theta}(z | x)."""

    def __init__(self, arch: arch_t, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        MLP.__init__(self, arch, device, dtype)
        abc.ABC.__init__(self)

    @abc.abstractmethod
    def distribution(self, x: Tensor) -> Distribution:
        raise NotImplementedError


class GeneratorNet(MLP, abc.ABC):
    """Abstract generation network; implementing p_{theta}(y | z, x)."""

    def __init__(self, arch: arch_t, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        MLP.__init__(self, arch, device, dtype)
        abc.ABC.__init__(self)

    @abc.abstractmethod
    def distribution(self, z: Tensor, x: Tensor) -> Distribution:
        raise NotImplementedError
