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
import abc
import logging
import torch as t
import torch.nn as nn
import torch.nn.functional as F

import agnfinder.config as cfg

from typing import Optional, Callable, Any, Type
from torch.utils.data import DataLoader
from torchvision import transforms

from agnfinder.types import Tensor
from agnfinder.inference import utils


# Likelihoods -----------------------------------------------------------------


class SAN_Likelihood(object):

    def __init__(*args, **kwargs):
        pass

    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name of this distribution, as a string."""
        raise NotImplementedError

    @abc.abstractmethod
    def n_params(self) -> int:
        """Returns the number of parameters required to parametrise this
        distribution."""
        return 2

    @abc.abstractmethod
    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        """Evaluate the log probability of `value` under a distribution
        parametrised by `params`"""
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, params: Tensor) -> Tensor:
        """Draw a single sample from a distribution parametrised by `params`"""
        raise NotImplementedError


class Gaussian(SAN_Likelihood):

    def name(self) -> str:
        return "Gaussian"

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

    def name(self) -> str:
        return "Laplace"

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

    def name(self) -> str:
        return "MoG"

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

    def name(self) -> str:
        return "MoST"

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


class SAN(nn.Module):
    def __init__(self, cond_dim: int, data_dim: int, module_shape: list[int],
                 sequence_features: int, likelihood: Type[SAN_Likelihood],
                 likelihood_kwargs: Optional[dict[str, Any]] = None,
                 batch_norm: bool = False, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float32,
                 logging_callbacks: list[Callable] = [],
                 overwrite_results: bool = True) -> None:
        """"Sequential autoregressive network" implementation.

        Args:
            cond_dim: dimensions of the conditioning data (e.g. photometry)
            data_dim: dimensions of the data whose (conditional) distribution
                we are trying to learn (e.g. physical galaxy parameters)

            module_shape: widths of hidden 'module' layers
            sequence_features: the number of features to carry through between
                sequential blocks.
            likelihood: the likelihood to use for each p(y_d | y_<d, x)
            likelihood_kwargs: any parameters to pass to the likelihood's
                constructor
            batch_norm: whether to use batch normalisation in the network
            device: device memory to use
            dtype: datatype to use; defaults to float32 but float64 should be
                used if you have issues with numerical stability.
            logging_callbacks: list of callables accepting this model instance;
                often used for visualisations and debugging.
            overwrite_results: whether to overwrite a model with the same
                filepath (read, same parameters) at the ned of training. Default:
                True
        """

        super().__init__()

        self.cond_dim = cond_dim
        self.data_dim = data_dim
        self.module_shape = module_shape
        self.sequence_features = sequence_features

        kwargs = likelihood_kwargs if likelihood_kwargs is not None else {}
        self.likelihood: SAN_Likelihood = likelihood(**kwargs)

        self.dtype = dtype
        self.device = device
        self.batch_norm = batch_norm

        # initialise the network
        self.network_blocks = nn.ModuleList()
        self.block_heads = nn.ModuleList()

        for d in range(data_dim):
            b, h = self._sequential_block(cond_dim, d, module_shape,
                      out_shapes=[sequence_features, self.likelihood.n_params()],
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
        self.is_trained = False
        self.overwrite_results = overwrite_results
        self.savepath_cached: Optional[str] = None
        self.logging_callbacks = logging_callbacks

    def __repr__(self) -> str:
        return (f'SAN with {self.likelihood.name()} likelihood, '
                f'module blocks of shape {self.module_shape} '
                f'and {self.sequence_features} features between blocks')

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

    def trainmodel(self, train_loader: DataLoader, epochs: int = 5,
                   log_every: int= 1000) -> None:
        """Train the SAN model

        Args:
            train_loader: DataLoader to load the training data.
            epochs: number of epochs to train the model for
            log_every: logging frequency
        """

        self.train()

        for e in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                x, y = self.preprocess(x, y)

                # The following implicitly updates self.last_params, and
                # returns y_hat (a sample from p(y | x))
                _ = self.forward(x)

                # minimise NLL of the true ys using training parameters
                LP = self.likelihood.log_prob(y, self.last_params)
                loss = -LP.sum(1).mean(0)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if i % log_every == 0 or i == len(train_loader)-1:
                    # Run through all logging functions
                    [cb(self) for cb in self.logging_callbacks]
                    logging.info(
                        "Epoch: {:02d}/{:02d}, Batch: {:03d}/{:d}, Loss {:9.4f}"
                        .format(e+1, epochs, i, len(train_loader)-1, loss.item()))

        self.is_trained = True

    def sample(self, x: Tensor, n_samples = 1000) -> Tensor:
        """A convenience method for drawing (conditional) samples from p(y | x)
        for a single conditioning point.

        Args:
            cond_data: the conditioning data; x
            n_samples: the number of samples to draw

        Returns:
            Tensor: a tensor of shape [n_samples, data_dim]
        """
        x = x.unsqueeze(0) if x.dim() == 1 else x
        x, _ = self.preprocess(x, t.empty(x.shape))
        return self.forward(x.repeat_interleave(n_samples, 0))

    def fpath(self) -> str:
        """Returns a file path to save the model to, based on its parameters."""
        if self.savepath_cached is None:
            base = './results/sanmodels/'
            s = self.module_shape + [self.sequence_features]
            ms = '_'.join([str(l) for l in s])
            name = (f'l{self.likelihood.name()}_cd{self.cond_dim}'
                    f'_dd{self.data_dim}_ms{ms}_'
                    f'lp{self.likelihood.n_params()}_bn{self.batch_norm}')
            if self.overwrite_results:
                self.savepath_cached = f'{base}{name}.pt'
            else:
                n = 0
                while os.path.exists(f'{base}{name}_{n}.pt'):
                    n+=1
                self.savepath_cached = f'{base}{name}_{n}.pt'

        return self.savepath_cached


if __name__ == '__main__':

    import agnfinder.nbutils as nbu

    cfg.configure_logging()

    ip = cfg.InferenceParams()
    sp = cfg.SANParams()

    train_loader, test_loader = utils.load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=ip.batch_size,
        normalise_phot=utils.normalise_phot_np,
        transforms=[
            transforms.ToTensor()
        ])
    logging.info('Created data loaders')

    model = SAN(cond_dim=sp.cond_dim, data_dim=sp.data_dim,
                module_shape=sp.module_shape,
                sequence_features=sp.sequence_features,
                likelihood=sp.likelihood, batch_norm=True,
                likelihood_kwargs=sp.likelihood_kwargs,
                device=ip.device, dtype=ip.dtype,
                overwrite_results=ip.overwrite_results)

    if ip.device == t.device('cuda'):
        model.cuda()  # double check that everything is on GPU
    logging.info('Initialised SAN model')

    model.trainmodel(train_loader, ip.epochs)
    logging.info('Trained SAN model')

    t.save(model, model.fpath())
    logging.info(f'Saved SAN model as: {model.fpath()}')

    x, _ = nbu.new_sample(test_loader)
    model.sample(x, n_samples=1000)
    logging.info('Successfully sampled from model')
