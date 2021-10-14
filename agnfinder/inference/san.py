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

from typing import Optional, Callable, Any
from torch.utils.data import DataLoader
from torchvision import transforms

from agnfinder.types import Tensor
from agnfinder.inference import utils

class SAN(nn.Module):
    def __init__(self, cond_dim: int, data_dim: int, module_shape: list[int],
                 batch_norm: bool = False, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float32,
                 logging_callbacks: list[Callable] = [],
                 overwrite_results: bool = True) -> None:
        """"Sequential Autoregressive Network" implementation.

        Each dimension is currently fixed to a Gaussian likelihood;

            p(y_d | y_<d, x) ~ N(mu_yd, sigma_yd).

        Args:
            cond_dim: dimension of the conditioning data (e.g. photometry)
            data_dim: dimesions of the data whose (conditional) distribution we
                are trying to learn (e.g. physical galaxy parameters)
            module_shape: widths of hidden 'module' layers
            batch_norm: whether to use batch normalisation in the network
            device: device memory to use
            dtype: datatype to use; defaults to float32 but float64 should be
                used if you have issues with numerial stability.
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

        self.dtype = dtype
        self.device = device
        self.batch_norm = batch_norm

        self.network_blocks = nn.ModuleList()
        self.block_heads = nn.ModuleList()

        for d in range(data_dim):
            b, h = self._sequential_block(cond_dim, d, module_shape,
                                          out_shapes=[cond_dim, self.lparams],
                                          out_activations=[nn.ReLU, None])
            self.network_blocks.append(b)
            self.block_heads.append(h)

        # A tensor storing the parameters of the distributions from which
        # samples were drawn for the last forward pass.
        # Useful for evaluating NLL of data under model.
        # Size: [mini-batch, likelihood_params]
        self.last_pass_params: Optional[Tensor] = None

        self.opt = t.optim.Adam(self.parameters(), 1e-3)
        self.is_trained = False
        self.overwrite_results = overwrite_results
        self.savepath_cached: Optional[str] = None
        self.logging_callbacks = logging_callbacks

    def __repr__(self) -> str:
        return (f'SAN with {self.likelihood_name} likelihood and module '
                f'blocks of shape: {self.module_shape}')

    def _sequential_block(self, cond_dim: int, ctx: int, module_shape: list[int],
                          out_shapes: list[int],
                          out_activations: list[Any]
                          ) -> tuple[nn.Module, nn.ModuleList]:

        assert len(out_shapes) == len(out_activations)

        block = nn.Sequential()
        heads = nn.ModuleList()

        hs: list[int] = [cond_dim + ctx] + module_shape
        for i, (j, k) in enumerate(zip(hs[:-1], hs[1:])):
            block.add_module(name=f'B{ctx}L{i}', module=nn.Linear(j, k))
            if self.batch_norm:
                block.add_module(name=f'B{ctx}BN{i}', module=nn.BatchNorm1d(k))
            block.add_module(name=f'B{ctx}A{i}', module=nn.ReLU())
        block.to(self.device, self.dtype)

        hn: int = module_shape[-1]
        for i, h in enumerate(out_shapes):
            this_head = nn.Sequential()
            this_head.add_module(name=f'B{ctx}H{i}', module=nn.Linear(hn, h))

            a = out_activations[i]
            if a is not None:
                this_head.add_module(name=f'B{ctx}A{i}', module=a())
            heads.append(this_head)
        heads.to(self.device, self.dtype)

        return block, heads

    @property
    def likelihood_name(self) -> str:
        """Name of the likelihood function"""
        raise NotImplementedError

    @property
    def lparams(self) -> int:
        """Gives the number of parameters to return for each likelihood
        p(y_d | y_<d, x)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rsample_likelihood(self, lparams: Tensor) -> Tensor:
        """Computes a reparametrised sample from the likelihood density, using
        the provided likelihood parameters `lparams`.

        Args:
            lparams: likelihood parameters, returned from a sequential block
                (shape [min-batch, N]) where N is e.g. 2 for a Gaussian.

        Returns:
            Tensor: reparametrised sample of shape [mini-batch, 1].
        """
        B = lparams.size(0)
        loc, scale = lparams.split(1, 1)
        rsamples = loc + utils.squareplus_f(scale) * \
                t.randn((B, 1), dtype=self.dtype, device=self.device)
        return rsamples

    @abc.abstractmethod
    def eval_log_likelihood(self, y: Tensor, lparams: Tensor) -> Tensor:
        """Evaluate the likelihood of y under the likelihood parametrised by
        lparams.

        Args:
            y: tensor of shape [mini-batch, data_dim] the likelihood of which
                to evaluate.
            lparams: parameters for the likelihood, of size
                [mini-batch, data_dim, N] where N is the number of parameters
                required for the chosen distribution (e.g. N=2 for univariate
                Gaussian)
        Returns:
            Tensor: a tensor of shape [mini-batch, 1]
        """
        loc, scale = lparams.split(1, -1)
        loc, scale = loc.squeeze(), utils.squareplus_f(scale.squeeze())
        loss = F.gaussian_nll_loss(y, loc, scale, reduction='none')
        return -loss.sum(1)


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

        block_input = x
        data = t.empty((B, 0), dtype=self.dtype, device=self.device)

        self.last_params = t.empty((B, self.data_dim, self.lparams),
                                   dtype=self.dtype, device=self.device)

        for d in range(self.data_dim):

            ctx_input = t.cat((block_input, data), 1)
            H = self.network_blocks[d](ctx_input)

            # update block_input ready for next layer
            block_input = self.block_heads[d][0](H)
            # get the 'likelihood parameters'
            l_params = self.block_heads[d][1](H)

            # reparametrised sampling from likelihood
            y_d = self.rsample_likelihood(l_params)

            data = t.cat((data, y_d), -1)

            self.last_params[:, d] = l_params

        assert data.shape == (x.size(0), self.data_dim)
        return data

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
                   log_every: int = 500) -> None:
        """Train the SAN model.

        Args:
            train_loader: DataLoader to load the training data
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

                # TODO:
                # Would also be interesting to try to minimise something like
                # the MSE between the true y value and y_hat.

                # Minimise NLL of the true ys using the training parameters.
                l = self.eval_log_likelihood(y, self.last_params)
                loss = -l.mean(0)

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
        """A convenience method for drawing (conditional) samples from p(y | x).

        Args:
            cond_data: the conditioning data; x
            n_samples: the number of samples to draw

        Returns:
            Tensor: a tensor of shape [n_samples, data_dim]
        """
        x, _ = self.preprocess(x, t.empty(x.shape))
        return self.forward(x.expand(n_samples, *x.shape))

    def fpath(self) -> str:
        """Returns a file path to save the model to, based on its parameters."""
        if self.savepath_cached is None:
            base = './results/sanmodels/'
            ms = '_'.join([str(l) for l in self.module_shape])
            name = (f'l{self.likelihood_name}_cd{self.cond_dim}'
                    f'_dd{self.data_dim}_ms{ms}_'
                    f'lp{self.lparams}_bn{self.batch_norm}')
            if self.overwrite_results:
                self.savepath_cached = f'{base}{name}.pt'
            else:
                n = 0
                while os.path.exists(f'{base}{name}_{n}.pt'):
                    n+=1
                self.savepath_cached = f'{base}{name}_{n}.pt'

        return self.savepath_cached

class Gaussian_SAN(SAN):

    likelihood_name = "Gaussian"
    lparams = 2

    def rsample_likelihood(self, lparams: Tensor) -> Tensor:
        B = lparams.size(0)
        loc, scale = lparams.split(1, 1)
        rsamples = loc + utils.squareplus_f(scale) * \
                t.randn((B, 1), dtype=self.dtype, device=self.device)
        return rsamples

    def eval_log_likelihood(self, y: Tensor, lparams: Tensor) -> Tensor:
        loc, scale = lparams.split(1, -1)
        loc, scale = loc.squeeze(), utils.squareplus_f(scale.squeeze())
        loss = -F.gaussian_nll_loss(y, loc, scale, reduction='none').sum(1)
        return loss
        # loss = 0.5 * (t.log(scale) + (y - loc)**2 / scale)
        # return loss.sum(1)

class Laplace_SAN(SAN):

    likelihood_name = "Laplace"
    lparams = 2

    def rsample_likelihood(self, lparams: Tensor) -> Tensor:
        B = lparams.size(0)  # batch size
        loc, scale = lparams.split(1, 1)
        scale = utils.squareplus_f(scale)
        finfo = t.finfo(loc.dtype)
        u = loc.new(t.Size((B,1))).uniform_(finfo.eps - 1, 1)
        return loc - scale * u.sign() * t.log1p(-u.abs())

    def eval_log_likelihood(self, y: Tensor, lparams: Tensor) -> Tensor:
        loc, scale = lparams.split(1, -1)
        loc, scale = loc.squeeze(), utils.squareplus_f(scale.squeeze())
        loss = -t.log(2*scale) - t.abs(y - loc) / scale
        return loss.sum(1)

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

    # Construct the model
    # model = Gaussian_SAN(sp.cond_dim, sp.data_dim, sp.module_shape,
    #                      device=ip.device, dtype=ip.dtype)
    model = Laplace_SAN(sp.cond_dim, sp.data_dim, sp.module_shape,
                        device=ip.device, dtype=ip.dtype)
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
