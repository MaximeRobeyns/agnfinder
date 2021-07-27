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

from typing import Union, Callable
from prospect.models import priors

# Type for the limits on the free parameters.
paramspace_t = dict[str, tuple[float, float]]

# Type for CPz model parameter description
pdict_t = dict[str, Union[float, bool, str, priors.Prior]]


# Maybe 'monad' ---------------------------------------------------------------
# Monad for safer and more explicit CPz model parameter definitions.

class MaybeFloat():
    def __init__(self):
        self.value = None

    # kind of inspired by Haskell's bind (>>)
    def use(self, free_action: Callable[[], pdict_t],
                  just_action: Callable[[float], pdict_t]):
        if self.__class__ == _Maybe__Free:
            return free_action()
        elif self.__class__ == Just:
            assert isinstance(self.value, float)
            return just_action(self.value)

class _Maybe__Free(MaybeFloat):
    def __repr__(self):
        return "Free"

Free = _Maybe__Free()

class Just(MaybeFloat):
    def __init__(self, v: float):
        self.value: float = v

    def __repr__(self):
        return f'Just({self.value})'


# Optional ---------------------------------------------------------------------
# Allows for optional parameters. This is really just another Maybe data type,
# but I have called it 'Optional' for better clarity

class Optional():
    def __init__(self):
        self.value = None

    def use(self, nothing_action: Callable[[], None],
                  free_action: Callable[[], pdict_t],
                  just_action: Callable[[float], pdict_t]):
        if self.__class__ == _Optional__Left:
            return nothing_action()
        elif self.__class__ == OptionalValue:
            assert isinstance(self.value, MaybeFloat)
            return self.value.use(free_action, just_action)

class _Optional__Left(Optional):
    def __repr__(self):
        return "Nothing"

Nothing = _Optional__Left()

# The _Optional__Right value; called OptionalValue for syntactic simplicity
class OptionalValue(Optional):
    def __init__(self, v: MaybeFloat):
        self.value = v

    def __repr__(self):
        return f'OptionalValue({repr(self.value)})'
