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
"""Create and load quasar templates."""

import abc
import dill
import logging

import numpy as np
import pandas as pd


class InterpolatedTemplate(metaclass=abc.ABCMeta):

    def __init__(self, template_loc: str, data_loc: str = ""):
        self.data_loc: str = data_loc
        self.template_loc: str = template_loc

        if not self.data_loc == "":
            logging.warning('Creating new template - \
                    will overwrite existing templates on disk')
            self._interpolated_template = self._create_template()
            self._save_template()
        else:
            self._interpolated_template = self._load_template()

    @abc.abstractmethod
    def _create_template(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _eval_template(self):
        raise NotImplementedError

    def _save_template(self):
        with open(self.template_loc, 'wb') as f:
            dill.dump(self._interpolated_template, f)

    def _load_template(self):
        with open(self.template_loc, 'rb') as f:
            return dill.load(f)

    def __call__(self, *args, **kwargs):
        return self._eval_template(*args, **kwargs)


class QuasarTemplate(InterpolatedTemplate):

    def _eval_template(self, wavelengths: np.ndarray,
                       short_only: bool = False) -> np.ndarray:
        fluxes = 10**self._interpolated_template(np.log10(wavelengths))
        if short_only:
            # TODO implement this
            fluxes *= get_damping_multiplier(wavelengths, 'long')
        return fluxes

    def _create_template(self) -> np.ndarray:

        # radio-quiet mean quasar template from
        # https://iopscience.iop.org/article/10.1088/0067-0049/196/1/2#apjs400220f6.tar.gz

        df = pd.read_csv(self.data_loc, skiprows=19, delim_whitespace=True).read()
        assert isinstance(df, pd.DataFrame)
        log_freq = df['log_freq']
        assert log_freq is not None
        freqs = 10**log_freq.to_numpy()

        wavelengths = 299792458. / freqs  # 'Research code' ladies and gentlemen!

        # interpolate in log wavelength(A)/log freq(Hz) space
        interp = interp1d(
            np.log10(wavelengths * 1e10),  # in angstroms
            df['log_flux'],
            kind='linear'
        )
        # TODO implement this
        normalised_interp = normalise_template(interp)
        return normalised_interp


