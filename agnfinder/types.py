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
""" Some custom types """

import typing
import numpy as np
import torch as t
import torch.nn as nn

from sedpy import observate
from prospect.models import priors
from typing import Union, Callable, Any

from agnfinder.utils import ConfigClass

# Type for the limits on the free parameters.
paramspace_t = dict[str, tuple[float, float]]

# Tyep for CPz observation dictionary
# - Shame that we have to use `Any` here, but this is due to Prospector using
#   an arbitrary dictionary of params.
cpz_obs_dict_t = dict[str, Union[np.ndarray, list[observate.Filter], Any]]

# Type for CPz model parameter description
pdict_t = dict[str, Union[float, bool, str, priors.Prior]]

# Type for prospector run parameters
prun_params_t = dict[str, Union[int, bool, float, None]]


# Neural network related ------------------------------------------------------


# A PyTorch Tensor
Tensor = t.Tensor
# One or more tensors used to parametrise a distribution
DistParams = list[Tensor]
# NumPy array or PyTorch tensor
tensor_like = Union[np.ndarray, Tensor]


# Filters ---------------------------------------------------------------------

# 'Enum' for filter selection
class FilterSet():
    def __init__(self, name: str, dim: int):
        self.value = name
        self.dim = dim
    def __repr__(self) -> str:
        return self.value
    def __eq__(self, other) -> bool:
        return self.value == other

class _Euclid(FilterSet): pass
class _Reliable(FilterSet): pass
class _All(FilterSet): pass

class Filters():
    Euclid = _Euclid('euclid', 8)
    Reliable = _Reliable('reliable', 12)
    All = _All('all', 12)


# Maybe 'monad' ---------------------------------------------------------------
# Monad for safer and more explicit CPz model parameter definitions.


class MaybeFloat():
    def __init__(self):
        self.value: Union[float, bool] = False

    # kind of inspired by Haskell's bind (>>)
    def use(self, free_action: Callable[[], pdict_t],
                  just_action: Callable[[float], pdict_t]):
        if self.__class__ == _Maybe__Free:
            return free_action()
        elif self.__class__ == Just:
            assert isinstance(self.value, float)
            return just_action(self.value)


class _Maybe__Free(MaybeFloat):
    def __init__(self):
        self.value: bool = True

    def __repr__(self):
        return "Free"

Free = _Maybe__Free()


class Just(MaybeFloat):
    def __init__(self, v: float):
        assert isinstance(v, float)  # type annotations aren't enforced :(
        self.value: float = v

    def __repr__(self):
        return f'Just({self.value})'


# Optional ---------------------------------------------------------------------
# Allows for optional parameters. This is really just another Maybe data type,
# but I have called it 'Optional' for ease of use


class Optional():
    def __init__(self):
        self.value = None
        self.maybefloatvalue = None

    def use(self, nothing_action: Callable[[], None],
                  free_action: Callable[[], pdict_t],
                  just_action: Callable[[float], pdict_t]):

        if self.__class__ == _Optional__Left:
            return nothing_action()
        elif self.__class__ == OptionalValue:
            assert isinstance(self.maybefloatvalue, MaybeFloat)
            return self.maybefloatvalue.use(free_action, just_action)

class _Optional__Left(Optional):
    def __init__(self):
        super().__init__()
        self.value = False

    def __repr__(self):
        return "Nothing"

Nothing = _Optional__Left()

# The _Optional__Right value; called OptionalValue for syntactic simplicity
class OptionalValue(Optional):
    def __init__(self, v: MaybeFloat):
        self.value: Union[bool, float] = v.value
        self.maybefloatvalue: MaybeFloat = v

    def __repr__(self):
        return f'OptionalValue({repr(self.maybefloatvalue)})'


# Free Parameter Configuration ------------------------------------------------


column_order: list[str] = ['redshift', 'log_mass', 'dust2', 'tage', 'log_tau',
        'log_agn_mass', 'agn_eb_v', 'log_agn_torus_mass', 'inclination']


class FreeParameters(ConfigClass):

    def __init__(self):

        # This list provides the order for the physical parameters.
        self.raw_members: list[str] = column_order

        self.raw_params: paramspace_t = {}

        tmp_logs: list[bool] = []
        tmp_params: list[tuple[float, float]] = []

        # fill in all the parameter values.
        for m in self.raw_members:
            self.raw_params[m] = getattr(self, m)
            tmp_logs.append(True if m.startswith('log') else False)
            tmp_params.append(getattr(self, m))

        self.log = t.tensor(tmp_logs, dtype=t.bool)
        self.params = t.tensor(tmp_params, dtype=t.float64)

    def __len__(self) -> int:
        return len(self.raw_params)

    @property
    def redshift(self) -> tuple[float, float]:
        raise NotImplementedError

    @property
    def log_mass(self) -> tuple[float, float]:
        raise NotImplementedError

    @property
    def log_agn_mass(self) -> tuple[float, float]:
        raise NotImplementedError

    @property
    def log_agn_torus_mass(self) -> tuple[float, float]:
        raise NotImplementedError

    @property
    def dust2(self) -> tuple[float, float]:
        raise NotImplementedError

    @property
    def tage(self) -> tuple[float, float]:
        raise NotImplementedError

    @property
    def log_tau(self) -> tuple[float, float]:
        raise NotImplementedError

    @property
    def agn_eb_v(self) -> tuple[float, float]:
        raise NotImplementedError

    @property
    def inclination(self) -> tuple[float, float]:
        raise NotImplementedError


# Feed-Forward MLP Architecture Description -----------------------------------


class arch_t(ConfigClass):

    def __init__(self, layer_sizes: list[int], head_sizes: list[int],
                 activations: Union[nn.Module, list[nn.Module]],
                 head_activations: typing.Optional[
                     list[typing.Optional[nn.Module]]] = None,
                 batch_norm: bool = True):
        """A class to describe a (non-convolutional) MLP architecture.

        Args:
            layer_sizes: size of input, and hidden layers.
            head_sizes: size of output layer(s)
            activations: instances of activation functions to apply to input /
                hidden layers. The same activation is re-used for all layers if
                this is not a list.
            head_activations: Optional list of activation functions to apply to
                outputs. Can be ``None``, or a list of optional instances of
                activation functions.
            batch_norm: whether to apply batch normalisation at each layer.

        Raises:
            ValueError: if ``layer_sizes`` is an empty list (minimum: one input
                layer)
            ValueError: if ``len(layer_sizes) != len(activations)`` when
                activations is a list
            ValueError: if an activation function does not extend nn.Module
            ValueError: if ``head_sizes`` is not ``list[int]`` of length at
                least one
            ValueError: if ``len(head_sizes) != len(head_activations)``

        Examples:

            ANN with 1 hidden layer, ReLU activations, no batch normalisation,
            and 2 output heads with different activation functions

            >>> arch_t(layer_sizes=[28*28, 256], head_sizes=[10, 2], \
            ...        activations=nn.ReLU(), \
            ...        head_activations=[nn.Softmax(), nn.ReLU()] \
            ...        batch_norm=False)

            ANN with 1 hidden layer, ReLU activations, no output activation &
            batch normalisation:

            >>> arch_t([512, 256], [10], nn.ReLU())

            ANN with two output heads, one without and one with activation,
            respectively:

            >>> arch_t([2**i for i in range(10, 5, -1)], [10, 2], \
            ...        activations=nn.ReLU(),
            ...        head_activations=[None, nn.Softmax()])

        """

        if len(layer_sizes) == 0:
            raise ValueError((
                'layer_sizes cannot be an empty list: at the very least, '
                'the input shape must be specified.'))

        self._activations: list[nn.Module] = []
        self._layer_sizes: list[int] = layer_sizes
        self._head_sizes: list[int] = head_sizes
        self._batch_norm: bool = batch_norm

        if isinstance(activations, list):
            if not len(layer_sizes) == len(activations):
                raise ValueError((
                    f'Number of activation functions ({len(activations)}) does '
                    f'not match number of layers ({len(layer_sizes)}).'))
            if not all([isinstance(a, nn.Module) for a in activations]):
                raise ValueError("activations must extend nn.Module")
            self._activations = activations
        else:
            if not isinstance(activations, nn.Module):
                raise ValueError("activations must extend nn.Module")
            self._activations = len(layer_sizes) * [type(activations)()]

        if head_activations is None:
            self._head_activations: list[Union[None, nn.Module]] = \
                [None for _ in head_sizes]
        else:
            if not isinstance(head_activations, list):
                raise ValueError("head_activations must be a list")
            if not all([isinstance(a, nn.Module)
                        for a in head_activations if a is not None]):
                raise ValueError("head activations must extend nn.Module")
            if len(head_sizes) != len(head_activations):
                raise ValueError((
                    f'length of head_activations ({len(head_activations)})',
                    f'must equal length of head_sizes ({len(head_sizes)}))'))
            self._head_activations = head_activations

    @property
    def activations(self) -> list[nn.Module]:
        return self._activations

    @property
    def layer_sizes(self) -> list[int]:
        return self._layer_sizes

    @property
    def in_shape(self) -> int:
        return self._layer_sizes[0]

    @property
    def batch_norm(self) -> bool:
        return self._batch_norm

    @property
    def head_sizes(self) -> list[int]:
        return self._head_sizes

    @property
    def head_activations(self) -> list[Union[None, nn.Module]]:
        return self._head_activations

    def __len__(self) -> int:
        return len(self.layer_sizes) + 1


# SED Components --------------------------------------------------------------


class SEDComponents(object):

    def __init__(self, wavelengths: np.ndarray, galaxy: np.ndarray,
                 unextincted_quasar: np.ndarray, extincted_quasar: np.ndarray,
                 torus: np.ndarray, net: np.ndarray):

        self.wavelengths = wavelengths
        self.galaxy = galaxy
        self.unextincted_quasar = unextincted_quasar
        self.extincted_quasar = extincted_quasar
        self.torus = torus
        self.net = net

    def components(self) -> list[tuple[str, np.ndarray]]:
        return [
            ('wavelengths', self.wavelengths),
            ('galaxy', self.galaxy),
            ('unextincted_quasar', self.unextincted_quasar),
            ('extincted_quasar', self.extincted_quasar),
            ('torus', self.torus),
            ('net', self.net),
        ]
