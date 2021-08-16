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
""" Implements conditional VAE """

import torch as t
import torch.nn as nn

from agnfinder.types import arch_t


class CVAE(t.Module):

    # def __init__(self, enc_arch: arch_t, dec_arch: arch_t, latent_size: int,
    #              cond_size: int):
    #     """Conditional Variational AutoEncoder

    #     Args:
    #         enc_arch: encoder architecture
    #         dec_arch: decoder architecture
    #         latent_size: size of latent vector; must equal output of enc_arch
    #         cond_size: size of conditioning information.
    #     """
    #     if enc_arch.out_shape != latent_size:
    #         raise ValueError(('Latent vector size must be same dimension as '
    #                           'encoder output layer'))

    #     self.latent_size = latent_size
    #     # TODO configure which encoder is present here.
    #     self.encoder = Encoder(encoder_layers, latent_size, cond_size)


    def __init__(self, encoder_layers: list[int], latent_size: int,
                 cond_size: int, decoder_layers: list[int]):
        """Conditional Variational AutoEncoder

        Args:
            encoder_layers: list of encoder layer sizes
            latent_size: size of latent representation, z
            cond_size: size of conditioning information (representation)
            decoder_layers: list of decoder layer sizes

        Todo:
            Include architecture configuration information in custom type.
        """
        super().__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layers, latent_size, cond_size)
        self.decoder = Decoder(decoder_layers, latent_size, cond_size)

    def forward(self, y: t.Tensor, x: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
        """Compute reconstructed input y', given input y and conditioning
        information x

        Args:
            y: input
            x: conditioning information (available during training)

        Returns:
            tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]: reconstructed input,
                mean, log variance, latent.
        """

        means, log_var = self.encoder(y, x)
        # TODO perhaps also return the epsilon sample used for use in loss
        # function later, or pre-compute first term in q(z|y,x).
        z = self.reparametrise(means, log_var)
        out = self.decoder(z, x)

        return out, means, log_var, z

    def reparametrise(self, mu: t.Tensor, log_var: t.Tensor) -> t.Tensor:
        """
        Isotropic Gaussian reparametrisation
        """
        std = t.exp(0.5 * log_var)
        eps = t.randn_like(std)

        return mu + eps * std

    def inference(self, z: t.Tensor, x: t.Tensor) -> t.Tensor:
        """
        Just run the decoder, as if it was a discriminative model: y = f(z, x)
        """
        out = self.decoder(z, x)
        return out


class Encoder(nn.Module):

    def __init__(self, layer_sizes: list[int], latent_size: int,
                 cond_size: int):
        """ Encoder / recognition / inference network.

        Uses isotropic Gaussian z assumption.

        Args:
            encoder_layers: list of MLP layer sizes
            latent_size: size of latent representation, z
            cond_size: size of conditioning information representation
        """
        super().__init__()

        layer_sizes[0] += cond_size  # concatenate x and y for input

        self.MLP = nn.Sequential()
        for i, (j, k) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name=f'L{i}', module=nn.Linear(j, k))
            self.MLP.add_module(name=f'A{i}', module=nn.ReLU())

        self.mean = nn.Linear(layer_sizes[-1], latent_size)
        self.log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, y: t.Tensor, x: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        """Encode input x and conditioning information y to latent vector z

        Args:
            y: Input (i.e. photometric observations)
            x: Conditioning information (i.e. photometry)

        Returns:
            tuple[t.Tensor, t.Tensor]: Mean and diagonal log variance terms of
                isotropic Gaussian latent distribution p(z | x, y).
        """

        # TODO: perhaps implement the following for better combination of
        # conditioning information.
        # x = embed_conditioning_info(x)
        y = t.cat((y, x), -1)

        y = self.MLP(y)
        means = self.mean(y)
        log_vars = self.log_var(y)

        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, layer_sizes: list[int], latent_size: int,
            cond_size: int):
        """Decoder / generative network.

        Args:
            layer_sizes: list of MLP layer sizes
            latent_size: size of the latent representation, z
            cond_size: size of conditioning information
        """
        super().__init__()

        self.MLP = nn.Sequential()
        input_size = latent_size + cond_size

        for i, (j, k) in enumerate(zip([input_size]+layer_sizes[:-1],
                                       layer_sizes)):
            self.MLP.add_module(name=f'L{i}', module=nn.Linear(j, k))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name=f'A{i}', module=nn.ReLU())
            else:
                self.MLP.add_module(name=f'sigmoid', module=nn.Sigmoid())

    def forward(self, z: t.Tensor, x: t.Tensor) -> t.Tensor:
        """Evaluates p(y | z, x)

        Args:
            z: Latent vector
            x: Conditioning information (e.g. photometry)

        Returns:
            t.Tensor: likely y samples
        """

        # TODO: perhaps implement the following for better combination of
        # conditioning information.
        # x = embed_conditioning_info(x)
        z = t.cat((z, x), -1)

        y = self.MLP(z)

        return y
