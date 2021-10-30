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
Implements Masked Autoencoder for Distribution Estimation, Germain et al. 2015
https://arxiv.org/pdf/1502.03509.pdf with additional consideration given to
performing order / connectivity agnostic training efficiently on a GPU.

Note to reader: this file is left in for reference, but the MADE method was not
found to be terribly effective. This is why we make no reference to MADE in the
documentation and you are discouraged from using it.
"""

import os
import abc
import logging
import torch as t
import torch.nn as nn

from typing import Callable, Optional, Type, Any
from torchvision import transforms
from torch.utils.data import DataLoader
from abc import abstractmethod

from agnfinder.inference.inference import Model, ModelParams

from agnfinder import config as cfg
from agnfinder.types import Tensor
from agnfinder.inference import utils


# Likelihoods -----------------------------------------------------------------

class MADE_Likelihood(object):

    def __init__(self, data_dim: int, **_):
        self.data_dim = data_dim

    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name of this distribution, as a string"""
        raise NotImplementedError

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
    def sample(self, loc: Tensor, scale: Tensor) -> Tensor:
        """Draw a single sample from a distribution parametrised by `params`"""
        raise NotImplementedError


class Gaussian(MADE_Likelihood):

    def name(self) -> str:
        return "Gaussian"

    def _extract_params(self, params: Tensor) -> tuple[Tensor, Tensor]:
        loc, scale = params.split(self.data_dim, -1)
        return loc.squeeze(), utils.squareplus_f(scale).squeeze()

    def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
        return (t.distributions.Normal(*self._extract_params(params))
                               .log_prob(value).sum(1))

    def sample(self, loc: Tensor, scale: Tensor) -> Tensor:
        scale = utils.squareplus_f(scale)
        return t.distributions.Normal(loc, scale).sample()


# class Bernoulli(MADE_Likelihood):
#
#     def name(self) -> str:
#         return "Bernoulli"
#
#     def log_prob(self, value: Tensor, params: Tensor) -> Tensor:
#         raise NotImplementedError
#
#     def sapmle(self, params: Tensor) -> Tensor:
#         raise NotImplementedError


class MaskedLinear(nn.Linear):
    """Extends Linear by placing a configurable mask on the weights.

    Since we loop over multiple masks while training MADE in order to ensemble
    them, it is more efficient to store each mask (for this layer) in GPU
    memory, allowing a single forward pass to be parallelised across all the
    masks / connectivities, and the results averaged immediately for efficient
    ensembles on the GPU. This is why we store a `maskset` in this class, and
    perform a batch matrix multiplication `bmm` in the forward pass.
    """

    def __init__(self, in_features: int, out_features: int, maskset: Tensor,
                 bias: bool = True, device = None, dtype = None) -> None:
        """
        Args:
            maskset: The pre-computed masks for this layer.

        Note:
            By default, all masks are used. Call `update_mask_idxs` or provide
            a `mask_idxs` argument to `forward` to change this behaviour.
        """
        super().__init__(in_features, out_features, bias, device, dtype)
        # Save the masks in the state_dict, but not as trainable parameters.
        self.register_buffer('maskset', maskset)
        self.num_masks = maskset.size(0)  # M
        self.update_mask_idxs(t.arange(self.num_masks, device=maskset.device))

    def update_mask_idxs(self, idxs: Optional[Tensor] = None) -> None:
        """Selects a subset of the masks to use.

        Args:
            idxs: A 1D index list of masks to use. If set to None, all masks
                are used.
        """
        assert isinstance(self.maskset, Tensor)
        if idxs is None:
            # use the full set of masks if no mask indices are specified.
            self.masks = self.maskset
        else:
            self.masks = t.index_select(self.maskset, 0, idxs)

    def forward(self, X: Tensor) -> Tensor:
        """Performs a batch of linear transformations to the provided input `X`
        using the previously specified masks in the maskset. Averaging the
        outputs along the 0th dimension will provide an ensemble over the
        selected masks for implementing connectivity agnostic training.

        The linear transformation in question is :math:`y = x(A.M) + b`, where
        `.` is an element-wise product (masking) operation, where A is a matrix
        of weights, and M is a set of masking matrices.

        Let B be the number of masks / connectivities (len(mask_idxs)), N be
        the training mini-batch size (e.g. 1024), M be in_features and P be
        out_features, then this function f applies the transformation:

            f: [B, N, M] x [B, M, P] -> [B, N, P]

        Args:
            X: the input vector, of size [len(mask_idxs), N, in_features].

        Returns:
            Tensor: a tensor of size [N, len(mask_idxs), out_features]
        """
        B = len(self.masks)

        # transpose (due to bmm arguments) and allow for broadcasting
        MW = self.weight.T[None,:]

        # mask the weights element-wise
        MW = MW * self.masks
        assert MW.shape == (B, self.in_features, self.out_features)

        # expand the bias term to allow for broadcasting later
        eb = self.bias[None,None,:]

        # perform the batched linear transformation
        # y shape: [len(mask_idxs), X.size(1), out_features]
        y = t.bmm(X, MW) + eb
        assert y.shape == (len(self.masks), X.size(1), self.out_features)

        return y


# CMADE Description -----------------------------------------------------------


class MADEParams(ModelParams):
    """Configuration class for conditional MADE (CMADE).

    This defines some required properties, and additionally performs validation
    of user-supplied values.
    """

    def __init__(self):
        super().__init__()
        # perform any required validation here...

    @property
    @abstractmethod
    def hidden_sizes(self) -> list[int]:
        """Size of the hidden layers"""
        pass

    @property
    @abstractmethod
    def likelihood(self) -> Type[MADE_Likelihood]:
        """Likelihood to use for each p(y_d | y_<d, x)"""
        pass

    @property
    def likelihood_kwargs(self) -> Optional[dict[str, Any]]:
        """Any keyword arguments accepted by likelihood"""
        return None

    @property
    @abstractmethod
    def num_masks(self) -> int:
        """number of different orderings for order / connection agnostic training"""
        pass

    @property
    @abstractmethod
    def samples(self) -> int:
        """How many samples of connectivity / masks to average parameters over
        during inference"""
        pass

    @property
    @abstractmethod
    def natural_ordering(self) -> bool:
        """Whether to factor p(y | x) according to the natural order of y:
                p(y_1 | x) p(y_2 | y_1, x) p(y_3 | y_1, y_2, x) ...
        or not"""
        pass


class CMADE(Model):

    def __init__(self, mp: MADEParams,
                 logging_callbacks: list[Callable] = [],
                 overwrite_results: bool = True) -> None:
        """Conditional MADE implementation

        Args:
            mp: the model parameters, set in `config.py`.
            logging_callbacks: list of callables accepting this model instance;
                often used for visualisations and debugging.
            overwrite_results: whether to overwrite a model with the same
                filepath (read, same parameters) at the end of training. Default:
                True
        """
        super().__init__(mp, overwrite_results, logging_callbacks)

        self.hidden_sizes = mp.hidden_sizes

        kwargs = {} if mp.likelihood_kwargs is None else mp.likelihood_kwargs
        self.likelihood: MADE_Likelihood = mp.likelihood(**kwargs)

        self.out_size: int = self.data_dim * self.likelihood.n_params()

        # Setup GPU mask cache ================================================

        self.num_masks = mp.num_masks
        self.natural_ordering = mp.natural_ordering

        # Initialise a list of masks for each layer as well as a tensor of
        # num_masks orderings; stored in GPU memory
        self.masksets, self.orderings = self.initialise_masks()

        # Setup model =========================================================

        self.netlist = []
        self.hs: list[int] = [self.cond_dim + self.data_dim] + \
                             mp.hidden_sizes + [self.out_size]
        for i, (h0, h1) in enumerate(zip(self.hs, self.hs[1:])):
            self.netlist.extend([
                MaskedLinear(h0, h1, self.masksets[i]),
                nn.ReLU(),
            ])
        self.netlist.pop()  # pop the last activation for the output layer
        self.net: nn.Module = nn.Sequential(*self.netlist)
        self.net = self.net.to(device=self.device, dtype=self.dtype)

        # self.opt = t.optim.Adam(self.parameters(), 1e-3, weight_decay=1e-4)
        self.opt = t.optim.Adam(self.parameters(), 1e-3)

    def name(self) -> str:
        return f'{self.likelihood.name()}_MADE'

    def __repr__(self) -> str:
        return (f'Conditional MADE with {self.likelihood.name()} likelihood'
                f'ANN layer widths {self.hs}, a mask set of size '
                f'{self.num_masks}, {"no " if not self.natural_ordering else ""}'
                f'natural ordering, trained for {self.epochs} epochs with a '
                f'batch size of {self.batch_size}')

    def fpath(self) -> str:
        """Returns a file path to save the model to, based on parameters."""
        if self.savepath_cached == "":
            base = './results/mademodels/'
            hs = '_'.join([str(l) for l in self.hidden_sizes])
            no = 'T' if self.natural_ordering else 'F'
            name = (f'CMADE_l:{self.likelihood.name()}_h:{hs}_m:{self.num_masks}'
                    f'_nmasks:{self.num_masks}_no:{no}')
            if self.overwrite_results:
                self.savepath_cached = f'{base}{name}.pt'
            else:
                n = 0
                while os.path.exists(f'{base}{name}_{n}.pt'):
                    n+=1
                self.savepath_cached = f'{base}{name}_{n}.pt'
        return self.savepath_cached

    def update_mask_idxs(self, mask_idxs: Tensor = None) -> None:
        """Updates the masks that the network layers use to implement
        connectivity-agnostic training.

        Args:
            mask_idxs: 1D tensor of mask indexes. If set to None, all masks are
                used.
        """
        if mask_idxs is not None:
            mask_idxs = mask_idxs.to(device=self.device)
        for l in self.net.modules():
            if isinstance(l, MaskedLinear):
                l.update_mask_idxs(mask_idxs)

    def forward(self, xy: Tensor, mask_idxs: Tensor = None) -> Tensor:
        """
        Forward pass through the conditional masked autoencoder, with the
        conditioning data _prefixed_ to the data being modelled. That is,

        >>> xy = [cond_data, data]
        >>> xy.shape
        (N, cond_dim + data_dim)

        Args:
            xs: The input matrix, of size [mini_batch, cond_dim + data_dim]
            mask_idxs (optional): A 1D tensor of mask indices to use (see
                connectivity-agnostic training).

        Returns:
            A Tensor of shape [N, out_dim]
        """
        self.update_mask_idxs(mask_idxs)
        B = len(mask_idxs) if mask_idxs is not None else self.num_masks
        X = xy[None,:].expand(B, *xy.shape)

        # connectivity-agnostic ensemble is implemented b averaging over 0th dimension
        return self.net(X).mean(0)

    def alterantive_sample(self, x: Tensor, n_samples: int = 1000,
               mask_idxs: Tensor = None) -> Tensor:
        """An alternative variant of the sampling function which updates all
        dimensions of the output rather than just the dth one for each
        iteration of the sampling procedure through the data dimensions.
        """

        if mask_idxs is None:
            mask_idxs = t.arange(self.num_masks)
        self.update_mask_idxs(mask_idxs)

        B = mask_idxs.size(0)
        out = t.zeros((B, n_samples, self.data_dim),
                       dtype=self.dtype, device=self.device)

        expanded_x = x.expand(*out.shape[:2], x.size(1))
        selected_orderings = self.orderings.index_select(0, mask_idxs)

        for i in range(0, self.data_dim):
            X = t.cat((expanded_x, out), -1)
            net_outs = self.net(X)
            assert net_outs.shape == (B, n_samples, self.likelihood.n_params() * self.data_dim)

            # this is the modification
            cum_idxs = (selected_orderings <= i).nonzero(as_tuple=True)

            params: list[Tensor] = [net_outs[cum_idxs[0], :, cum_idxs[1] * p]
                                    for p in range(self.likelihood.n_params())]

            samples = self.likelihood.sample(*params)
            out[cum_idxs[0], :, cum_idxs[1]] = samples

        out = out.mean(0)
        return out

    def sample(self, x: Tensor, n_samples: int = 1000,
               *_, **kwargs) -> Tensor:
        """
        Draw a sample from p(y | x) by sequentially sampling
        y_1 | x, then y_2 | x, y_1, ..., until y_n | x, y_{<n}.

        This version is correct; it conditions correctly on the points, and
        samples in the appropriate order. However with each iteration of the
        sampling procedure, we keep the previously sampled points fixed, while
        adding new dimensions. This may result in a lack of cohesion, and this
        is what the alternative sampling method aims to address.

        Args:
            x: the conditioning information
            n_samples: the number of samples from p(y | x) to draw for each x.
            mask_idxs: the indexes of the masks / connectivities to ensemble
                each sample over.

        Returns:
            A [n_samples, data_dim] shaped tensor of samples.
        """
        mask_idxs: Tensor = kwargs['mask_idxs']

        if mask_idxs is None:
            # use all available masks
            mask_idxs = t.arange(self.num_masks)
        self.update_mask_idxs(mask_idxs)

        # number of masks
        B = mask_idxs.size(0)
        # doesn't actually matter what we initialise this to; zeros works
        out = t.zeros((B, n_samples, self.data_dim),
                       dtype=self.dtype, device=self.device)
        # expand x to shape [B, n_samples, cond_dim]
        expanded_x = x.expand(*out.shape[:2], x.size(1))
        # select the appropriate input orderings.
        selected_orderings = self.orderings.index_select(0, mask_idxs)

        # for each of the data dimensions
        for i in range(0, self.data_dim):
            # create network input by prepending conditioning information to output.
            X = t.cat((expanded_x, out), -1)

            net_outs = self.net(X)
            assert net_outs.shape == (B, n_samples, self.likelihood.n_params() * self.data_dim)
            idxs = (selected_orderings == i).nonzero(as_tuple=True)

            # idxs[0] give the mask indexes, idxs[1] give the orderings

            params: list[Tensor] = [net_outs[idxs[0], :, idxs[1] * p]
                                    for p in range(self.likelihood.n_params())]

            assert len(params) == self.likelihood.n_params()
            assert all([params[p].shape == (B, n_samples)
                        for p in range(self.likelihood.n_params())])

            samples = self.likelihood.sample(*params)
            assert samples.shape == (B, n_samples)

            out[idxs[0], :, idxs[1]] = samples

        # average across orderings and connectivities / masks
        out = out.mean(0)
        return out

    def initialise_masks(self) -> tuple[list[Tensor], Tensor]:
        """Builds a tensor of num_masks random (conditional) masks, stored in
        GPU memory.

        We return a separate 3D tensor for each layer of the feed-forward
        model, of dimension [num_masks, l[n]_dim, l[n-1]_dim], as well as
        another 2D tensor of shape [num_masks, data_dim] containing the
        factorisation orderings for each set of masks.
        """
        hs: list[int] = [self.cond_dim + self.data_dim] + self.hidden_sizes + \
                        [self.out_size]
        L = len(self.hidden_sizes)
        assert len(hs) == L+2

        # Begin by creating the random permutations for all layers ------------
        # Unfortunately, PyTorch doesn't offer vector operators for randperm or
        # randint, so we have to resort to looping over num_masks.
        #
        # This shouldn't be too problematic, since this is only performed once
        # when the model is first initialised (and not even again when loaded
        # from disk).

        # list of permutation matrices; one for each layer of the network,
        # where each row of a matrix corresponds to one mask sample.
        Ms: list[Tensor] = []
        if self.natural_ordering:
            Ms.append(t.arange(self.cond_dim + self.data_dim)
                       .repeat((self.num_masks, 1)))
        else:
            M0s: list[Tensor] = []
            for _ in range(self.num_masks):
                tmp_m = t.cat((
                    t.arange(self.cond_dim),
                    t.randperm(self.data_dim, requires_grad=False) + self.cond_dim
                    ), 0)[None,:]
                M0s.append(tmp_m)
            Ms.append(t.cat(M0s, 0))

        assert Ms[0].shape == (self.num_masks, self.cond_dim + self.data_dim)

        dd = self.cond_dim + self.data_dim - 1
        for l in range(1, L+1):
            Mls = []
            for p in Ms[l-1]:
                # TODO perhaps add cond_dim to the min here...
                Mls.append(t.randint(int(p.min().item()), dd,
                    (self.hidden_sizes[l-1],), requires_grad=False)[None,:])
            Ms.append(t.cat(Mls, 0))

        assert len(Ms) == len(hs) - 1

        # Now create the corresponding masks ----------------------------------

        masklist: list[Tensor] = []

        masklist.extend(
            [t.le(Ms[l].unsqueeze(-1), Ms[l+1].unsqueeze(1))
              .to(device=self.device, dtype=t.uint8)
            for l in range(0, L)])

        # Different connectivity constraints for output layer:
        outmask = t.lt(
            Ms[L].unsqueeze(-1), Ms[0][:,self.cond_dim:].unsqueeze(1)
        ).to(self.device, t.uint8)

        if self.likelihood.n_params() > 1:
            # repeat the final mask K times.
            outmask = outmask.repeat(1,1,self.likelihood.n_params())
        masklist.append(outmask.to(self.device, t.uint8))

        # TODO remove or move to test suite
        assert masklist[0].shape == (self.num_masks, hs[0], hs[1])
        for m in range(1, len(masklist)-1):
            assert masklist[m].shape == (self.num_masks, hs[m], hs[m+1])
        assert masklist[-1].shape == (
                self.num_masks, self.hidden_sizes[-1], self.out_size)
        assert len(masklist) == len(hs) - 1

        return masklist, (Ms[0] - self.cond_dim)[:,self.cond_dim:]

    def trainmodel(self, train_loader: DataLoader, ip: cfg.InferenceParams,
                   masks: Optional[int] = None) -> None:
        """Train the MADE model.

        Args:
            train_loader: DataLoader to load the training data.
            epochs: Number of epochs for which to train the model.
            masks: The number of samples of masks to average parameters over
                for each training iteration.
        """
        self.train()  # ensure that the model is in training mode.

        idxs: Optional[Tensor] = None

        for e in range(self.epochs):
            for i, (x, y) in enumerate(train_loader):

                if masks is not None:
                    idxs = t.randperm(self.num_masks)[:masks]

                # x is photometry, y are parameters (theta)
                x, y = self.preprocess(x, y)

                # concatenate x and y along rows
                xy = t.cat((x, y), 1)
                out = self.forward(xy, idxs)

                # Evaluate NLL loss
                # assume sum(1) already taken
                LP = self.likelihood.log_prob(y, out)

                loss = -LP.mean(0)

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


if __name__ == '__main__':
    import agnfinder.nbutils as nbu

    cfg.configure_logging()

    ip = cfg.InferenceParams()  # inference procedure parameters
    mp = cfg.MADEParams()  # MADE model parameters

    train_loader, test_loader = utils.load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=mp.batch_size,
        normalise_phot=utils.normalise_phot_np,
        transforms=[
            transforms.ToTensor()
        ])
    logging.info('Created data loaders')

    model = CMADE(mp)
    logging.info('Initialised Conditional MADE model')

    model.trainmodel(train_loader, ip)
    logging.info('Trained CMADE model')

    x, y = nbu.new_sample(test_loader)
    x, y = model.preprocess(x, y)
    x = x.unsqueeze(0)
    mask_idxs = t.randperm(model.num_masks)[:mp.samples]
    model.sample(x, n_samples=1, mask_idxs=mask_idxs)
    logging.info('Successfully sampled from model')
