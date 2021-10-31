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
"""
Implements a "sequential autoregressive network"; a simple, sequential
procedure for generating autoregressive samples.
"""

import os
import logging
import torch as t
import torch.nn as nn
import torch.nn.functional as F

import agnfinder.config as cfg

from abc import abstractmethod
from typing import Optional, Callable, Any, Type
from torch.utils.data import DataLoader
from torchvision import transforms

from agnfinder.types import Tensor
from agnfinder.inference import utils
from agnfinder.inference.inference import Model, ModelParams, InferenceParams


# Likelihoods -----------------------------------------------------------------


class SAN_Likelihood(object):

    def __init__(*args, **kwargs):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of this distribution, as a string."""
        raise NotImplementedError

    @abstractmethod
    def n_params(self) -> int:
        """Returns the number of parameters required to parametrise this
        distribution."""
        return 2

    @abstractmethod
    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        """Evaluate the log probability of `value` under a distribution
        parametrised by `params`"""
        raise NotImplementedError

    @abstractmethod
    def sample(self, params: Tensor) -> Tensor:
        """Draw a single sample from a distribution parametrised by `params`"""
        raise NotImplementedError


class Gaussian(SAN_Likelihood):

    name: str = "Gaussian"

    def _extract_params(self, params: Tensor) -> tuple[Tensor, Tensor]:
        loc, scale = params.split(1, -1)
        return loc.squeeze(), utils.squareplus_f(scale.squeeze())

    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        loc, scale = self._extract_params(params)
        return t.distributions.Normal(
                loc.squeeze(), scale.squeeze()).log_prob(value)

    def sample(self, params: Tensor) -> Tensor:
        return t.distributions.Normal(
                *self._extract_params(params)).sample()


class Laplace(SAN_Likelihood):

    name: str = "Laplace"

    def _extract_params(self, params: Tensor) -> tuple[Tensor, Tensor]:
        loc, scale = params.split(1, -1)
        return loc.squeeze(), utils.squareplus_f(scale.squeeze())

    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        return t.distributions.Laplace(
                *self._extract_params(params)).log_prob(value)

    def sample(self, params: Tensor) -> Tensor:
        return t.distributions.Laplace(
                *self._extract_params(params)).sample()


class MoG(SAN_Likelihood):

    def __init__(self, K: int) -> None:
        """Mixture of Gaussians.

        Args:
            K: number of mixture components.
        """
        self.K = K

    name: str = "MoG"

    def n_params(self) -> int:
        return 3 * self.K  # loc, scale, mixture weight.

    def _extract_params(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B = params.size(0)
        loc, scale, k = params.reshape(B, -1, self.K, 3).tensor_split(3, 3)
        return loc.squeeze(-1), utils.squareplus_f(scale).squeeze(-1), F.softmax(k, -1).squeeze(-1)

    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        loc, scale, k = self._extract_params(params)
        cat = t.distributions.Categorical(k)
        norms = t.distributions.Normal(loc, scale)
        return t.distributions.MixtureSameFamily(cat, norms).log_prob(value)

    def sample(self, params: Tensor) -> Tensor:
        B = params.size(0)
        loc, scale, k = params.reshape(B, self.K, 3).tensor_split(3, 2)
        loc, scale = loc.squeeze(-1), utils.squareplus_f(scale).squeeze(-1)
        cat = t.distributions.Categorical(F.softmax(k, -1).squeeze(-1))
        norms = t.distributions.Normal(loc, scale)
        return t.distributions.MixtureSameFamily(cat, norms).sample()

class MoST(SAN_Likelihood):

    def __init__(self, K: int) -> None:
        """Mixture of StudentT distributions.

        Args:
            K: number of mixture components.
        """
        self.K = K

    name: str =  "MoST"

    def n_params(self) -> int:
        return 3 * self.K  # loc, scale, mixture weight.

    def _extract_params(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B = params.size(0)
        loc, scale, k = params.reshape(B, -1, self.K, 3).tensor_split(3, 3)
        return loc.squeeze(-1), utils.squareplus_f(scale).squeeze(-1), F.softmax(k, -1).squeeze(-1)

    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        loc, scale, k = self._extract_params(params)
        cat = t.distributions.Categorical(k)
        sts = t.distributions.StudentT(1., loc, scale)
        return t.distributions.MixtureSameFamily(cat, sts).log_prob(value)

    def sample(self, params: Tensor) -> Tensor:
        B = params.size(0)
        loc, scale, k = params.reshape(B, self.K, 3).tensor_split(3, 2)
        loc, scale = loc.squeeze(-1), utils.squareplus_f(scale).squeeze(-1)
        cat = t.distributions.Categorical(F.softmax(k, -1).squeeze(-1))
        sts = t.distributions.StudentT(1., loc, scale)
        return t.distributions.MixtureSameFamily(cat, sts).sample()


# SAN Description -------------------------------------------------------------


class SANParams(ModelParams):
    """Configuration class for SAN.

    This defines some required properties, and additionally performs validation
    of user-supplied values.
    """

    def __init__(self):
        super().__init__()
        # perform any required validation here...

    @property
    @abstractmethod
    def module_shape(self) -> list[int]:
        """Size of each individual 'module block'"""
        pass

    @property
    @abstractmethod
    def sequence_features(self) -> int:
        """Number of features to carry through for each network block"""
        pass

    @property
    @abstractmethod
    def likelihood(self) -> Type[SAN_Likelihood]:
        """Likelihood to use for each p(y_d | y_<d, x)"""
        pass

    @property
    def likelihood_kwargs(self) -> Optional[dict[str, Any]]:
        """Any keyword arguments accepted by likelihood"""
        return None

    @property
    @abstractmethod
    def batch_norm(self) -> bool:
        """Whether to use batch normalisation or not"""
        pass


# Main SAN Class ==============================================================


class SAN(Model):
    def __init__(self, mp: SANParams,
                 logging_callbacks: list[Callable] = [],
                 overwrite_results: bool = True) -> None:
        """"Sequential autoregressive network" implementation.

        Args:
            mp: the model parameters, set in `config.py`.
            logging_callbacks: list of callables accepting this model instance;
                often used for visualisations and debugging.
            overwrite_results: whether to overwrite a model with the same
                filepath (read, same parameters) at the end of training. Default:
                True
        """
        super().__init__(mp, overwrite_results, logging_callbacks)

        self.module_shape = mp.module_shape
        self.sequence_features = mp.sequence_features

        kwargs = {} if mp.likelihood_kwargs is None else mp.likelihood_kwargs
        self.likelihood: SAN_Likelihood = mp.likelihood(**kwargs)

        self.batch_norm = mp.batch_norm

        # Initialise the network
        self.network_blocks = nn.ModuleList()
        self.block_heads = nn.ModuleList()

        for d in range(self.data_dim):
            b, h = self._sequential_block(self.cond_dim, d, self.module_shape,
                      out_shapes=[self.sequence_features,
                      self.likelihood.n_params()],
                      out_activations=[nn.ReLU, None])
            self.network_blocks.append(b)
            self.block_heads.append(h)

        # A tensor storing the parameters of the distributions from which
        # samples were drawn for the last forward pass.
        # Useful for evaluating NLL of data under model.
        # Size: [mini-batch, likelihood_params]
        # self.last_pass_params: Optional[Tensor] = None
        self.last_params: Optional[Tensor] = None

        self.opt = t.optim.Adam(self.parameters(), lr=1e-3)

        if mp.device == t.device('cuda'):
            self.to(mp.device, mp.dtype)

        # Strange mypy error requires this to be put here although it is
        # perfectly well defined and typed in the super class ¯\_(ツ)_/¯
        self.savepath_cached: str = ""

        self.name: str = f'{self.likelihood.name}_SAN'

    name: str = 'SAN'

    def __repr__(self) -> str:
        return (f'{self.name} with {self.likelihood.name} likelihood, '
                f'module blocks of shape {self.module_shape} '
                f'and {self.sequence_features} features between blocks trained '
                f'for {self.epochs} epochs with batches of size {self.batch_size}')

    def fpath(self) -> str:
        """Returns a file path to save the model to, based on its parameters."""
        if self.savepath_cached == "":
            base = './results/sanmodels/'
            s = self.module_shape + [self.sequence_features]
            ms = '_'.join([str(l) for l in s])
            name = (f'l{self.likelihood.name}_cd{self.cond_dim}'
                    f'_dd{self.data_dim}_ms{ms}_'
                    f'lp{self.likelihood.n_params()}_bn{self.batch_norm}'
                    f'_e{self.epochs}_bs{self.batch_size}')
            if self.overwrite_results:
                self.savepath_cached = f'{base}{name}.pt'
            else:
                n = 0
                while os.path.exists(f'{base}{name}_{n}.pt'):
                    n+=1
                self.savepath_cached = f'{base}{name}_{n}.pt'

        return self.savepath_cached

    def _sequential_block(self, cond_dim: int, d: int, module_shape: list[int],
                          out_shapes: list[int], out_activations: list[Any]
                          ) -> tuple[nn.Module, nn.ModuleList]:
        """Initialises a single 'sequential block' of the network.

        Args:
            cond_dim: dimension of conditioning data (e.g. photometric observations)
            d: current dimension in the autoregressive sequence p(y_d | y_<d, x)
            module_shape: sizes of the 'sequential block' layers
            out_shapes: size of sequential block output and parameters, respectively
            out_activations: activation functions to apply to each respective head

        Returns:
            tuple[nn.Module, nn.ModuleList]: sequential block and heads
        """

        block = nn.Sequential()
        heads = nn.ModuleList()

        if d == 0:
            hs = [cond_dim] + module_shape
        else:
            hs = [cond_dim + out_shapes[0] + d] + module_shape

        for i, (j, k) in enumerate(zip(hs[:-1], hs[1:])):
            block.add_module(name=f'B{d}L{i}', module=nn.Linear(j, k))
            if self.batch_norm:
                block.add_module(name=f'B{d}BN{i}', module=nn.BatchNorm1d(k))
            block.add_module(name=f'B{d}A{i}', module=nn.ReLU())
        block.to(self.device, self.dtype)

        hn: int = module_shape[-1]
        for i, h in enumerate(out_shapes):
            this_head = nn.Sequential()
            this_head.add_module(name=f'H{d}:{i}H{i}', module=nn.Linear(hn, h))

            a = out_activations[i]
            if a is not None:
                this_head.add_module(name=f'H{d}:{i}A{i}', module=a())
            heads.append(this_head)
        heads.to(self.device, self.dtype)

        return block, heads

    def forward(self, x: Tensor) -> Tensor:
        """Runs the autoregressive model.

        Args:
            x: some conditioning information

        Returns:
            Tensor: a sample from the distribution; y_hat ~ p(y | x)

        Implicit Returns:
            self.last_params: a tensor containing the parameters of each
                dimension's (univariate) distribution of size
                [mini-batch, lparams]; giving p(y | x)
        """

        # batch size
        B = x.size(0)
        ys = t.empty((B, 0), dtype=self.dtype, device=self.device)
        self.last_params = t.empty((B, self.data_dim, self.likelihood.n_params()),
                                   dtype=self.dtype, device=self.device)

        seq_features = t.empty((B, 0), dtype=self.dtype, device=self.device)

        for d in range(self.data_dim):

            d_input = t.cat((x, seq_features, ys), 1)

            H = self.network_blocks[d](d_input)

            # for passing to next sequential block
            seq_features = self.block_heads[d][0](H)

            # draw single sample from p(y_d | y_<d, x)
            params = self.block_heads[d][1](H)
            y_d = self.likelihood.sample(params).unsqueeze(1)

            ys = t.cat((ys, y_d), -1)
            self.last_params[:, d] = params

        # check we did the sampling right
        assert ys.shape == (x.size(0), self.data_dim)
        return ys

    def trainmodel(self, train_loader: DataLoader, ip: InferenceParams,
                   *args, **kwargs) -> None:
        """Train the SAN model

        Args:
            train_loader: DataLoader to load the training data.
            ip: The parameters to use for training, defined in
                `config.py:InferenceParams`.
        """
        self.train()

        for e in range(self.epochs):
            for i, (x, y) in enumerate(train_loader):
                x, y = self.preprocess(x, y)

                # The following implicitly updates self.last_params, and
                # returns y_hat (a sapmle from p(y | x))
                _ = self.forward(x)
                assert (self.last_params is not None)

                # Minimise the NLL of true ys using training parameters
                LP = self.likelihood.log_prob(y, self.last_params)
                loss = -LP.sum(1).mean(0)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if i % ip.logging_frequency == 0 or i == len(train_loader)-1:
                    # Run through all logging functions
                    [cb(self) for cb in self.logging_callbacks]
                    logging.info(
                        "Epoch: {:02d}/{:02d}, Batch: {:05d}/{:d}, Loss {:9.4f}"
                        .format(e+1, self.epochs, i, len(train_loader)-1,
                                loss.item()))

    def sample(self, x: Tensor, n_samples = 1000, *args, **kwargs) -> Tensor:
        """A convenience method for drawing (conditional) samples from p(y | x)
        for a single conditioning point.

        Args:
            cond_data: the conditioning data; x
            n_samples: the number of samples to draw
            kwargs: any additional model-specific arguments

        Returns:
            Tensor: a tensor of shape [n_samples, data_dim]
        """
        x = x.unsqueeze(0) if x.dim() == 1 else x
        x, _ = self.preprocess(x, t.empty(x.shape))
        return self.forward(x.repeat_interleave(n_samples, 0))


if __name__ == '__main__':

    import agnfinder.nbutils as nbu

    cfg.configure_logging()

    ip = cfg.InferenceParams()
    sp = cfg.SANParams()

    train_loader, test_loader = utils.load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=sp.batch_size,
        normalise_phot=utils.normalise_phot_np,
        transforms=[
            transforms.ToTensor()
        ])
    logging.info('Created data loaders')

    model = SAN(sp)
    logging.info('Initialised SAN model')

    # NOTE: uses cached model (if available), and saves to disk after training.
    model.trainmodel(train_loader, ip)
    logging.info('Trained SAN model')

    x, _ = nbu.new_sample(test_loader)
    model.sample(x, n_samples=1000)
    logging.info('Successfully sampled from model')
