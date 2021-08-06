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

import numpy as np

from sedpy import observate
from prospect.models import priors
from typing import Union, Callable, Any

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


# Type for class-based configuration objects ---------------------------------


class ConfigClass(object):
    """The point of this class is to allow class-based configuration objects to
    be printed for logging and debugging in a clean way.
    """

    def __repr__(self) -> str:
        r = f'\n\n{79*"="}\n'
        c = f'Configuration class `{type(self).__name__}`'
        n = len(c)
        nn = int((79 - n) / 2)
        r += nn * ' ' + c + f'\n{79*"-"}\n\n'
        members = [a for a in dir(self) if not callable(getattr(self, a))\
                   and not a.startswith("__")]
        for m in members:
            r += f'{m}: {getattr(self, m)}\n'
        r += '\n' + 79 * '=' + '\n\n'
        return r
