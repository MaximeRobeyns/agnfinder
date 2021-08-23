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

from typing import Union, Optional
from torch.utils.data import DataLoader

from agnfinder.types import Tensor, DistParams, arch_t, CVAEParams


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

    def forward(self, x: Tensor) -> Union[Tensor, DistParams]:
        """Forward pass through the network

        Args:
            x: input tensor

        Returns:
            Union[Tensor, DistParams]: Return single tensor if there is a single
                head, otherwise a list of tensors (DistParams) with each element
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


class _CVAE_Dist():
    """Base distribution class for use with encoder, decoder and prior"""

    def __init__(self, batch_shape=t.Size(), event_shape=t.Size(),
                 device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self.device: t.device = device
        self.dtype: t.dtype = dtype

    def __call__(self, *args, **kwargs):
        self.log_prob(*args, **kwargs)

    def _extended_shape(self, sample_shape=t.Size()):
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        Args:
            sample_shape (t.Size): the size of the sample to be drawn.
        """
        if not isinstance(sample_shape, t.Size):
            sample_shape = t.Size(sample_shape)
        return sample_shape + self._batch_shape + self._event_shape

    @abc.abstractmethod
    def log_prob(self, value: Tensor) -> Tensor:
        """
        Returns the log of the probability density/mass function evaluated at
        value.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        raise NotImplementedError


class _CVAE_RDist(_CVAE_Dist):
    """Explicit Rdist class for distributions implementing reparametrised
    sampling.
    """

    def sample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        return self.rsample(sample_shape)

    @abc.abstractmethod
    def rsample(self, sample_shape: t.Size = t.Size()) -> Tensor:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
        raise NotImplementedError


class CVAEPrior(MLP, abc.ABC):
    """Base prior class for CVAEs

    This implements the (optional) prior network which parametrises
    p_{theta}(z | x)
    """
    def __init__(self, arch: Optional[arch_t],
                 device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        if arch is not None:
            MLP.__init__(self, arch, device, dtype)
        abc.ABC.__init__(self)

    def __call__(self, x: Tensor) -> _CVAE_Dist:
        if self.MLP is not None:
            dist_params = self.forward(x)
            return self.get_dist(dist_params)
        return self.get_dist()

    @abc.abstractmethod
    def get_dist(self, dist_params: Optional[Union[Tensor, DistParams]] = None
                 ) -> _CVAE_Dist:
        raise NotImplementedError


class CVAEEnc(MLP, abc.ABC):
    """Base encoder class for CVAEs

    This implements the recognition network which parametrises
    q_{phi}(z | y, x)
    """

    def __init__(self, arch: arch_t, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        MLP.__init__(self, arch, device, dtype)
        abc.ABC.__init__(self)

    def __call__(self, y: Tensor, x: Tensor) -> _CVAE_RDist:
        tmp = t.cat((y, x), -1)
        dist_params = self.forward(tmp)
        return self.get_dist(dist_params)

    @abc.abstractmethod
    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_RDist:
        raise NotImplementedError


class CVAEDec(MLP, abc.ABC):
    """Base decoder class for CVAEs

    This implements the generator network which parametrises
    p_{theta}(y | z, x)
    """

    def __init__(self, arch: arch_t, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        MLP.__init__(self, arch, device, dtype)
        abc.ABC.__init__(self)

    @abc.abstractmethod
    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_Dist:
        raise NotImplementedError

    def __call__(self, z: Tensor, x: Tensor) -> _CVAE_Dist:
        tmp = t.cat((z, x), -1)
        dist_params = self.forward(tmp)
        return self.get_dist(dist_params)


class CVAE(nn.Module, abc.ABC):
    """Conditional VAE class

    You must provide the following distributions in the configuration.
    - q_{phi}(z | y, x)   approximate posterior / encoder
    - p_{theta}(z | x)    prior / encoder
    - p_{theta}(y | z, x) generator / decoder
    """

    def __init__(self, cp: CVAEParams,
                 device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        """Initialise a CVAE

        Args:
            cp: CVAE parameters (usually from config.py)
            device: device on which to store the model
            dtype: data type to use in Tensors
        """
        super().__init__()  # init nn.Module, ABC
        self.device = device
        self.dtype = dtype
        self.latent_dim: int = cp.latent_dim

        self.prior = cp.prior(cp.prior_arch, device, dtype)
        self.encoder = cp.encoder(cp.enc_arch, device, dtype)
        self.decoder = cp.decoder(cp.dec_arch, device, dtype)

        nets: list[MLP] = [self.prior, self.encoder, self.decoder]
        self.opt = t.optim.Adam([param for n in nets for param in n.parameters()])

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

    def trainmodel(self, train_loader: DataLoader, epochs: int = 10,
                   log_every: int = 100) -> None:
        """
        This is a potentially better way of organising the training process
        and encapsulating the functionality of the prior, encoder and
        decoder distributions for better modularity and separation of
        concerns.
        """
        b = train_loader.batch_size  # batch size
        assert isinstance(b, int)
        ipe = len(train_loader) * b  # iterations per epoch
        t = epochs * ipe

        for e in range(epochs):
            for i, (x, y) in enumerate(train_loader):

                x, y = self.preprocess(x, y)

                # Get q_{phi}(z | y, x)
                q = self.encoder(y, x)
                # Sample latent from q
                z = q.sample()

                # Get p_{theta}(z | x)
                pr = self.prior(x)

                # Get p_{theta}(y | z, x)
                p = self.decoder(z, x)

                logpy = p.log_prob(y)
                logpz = pr.log_prob(z)
                logqz = q.log_prob(z)

                ELBO = self.ELBO(logpy, logpz, logqz, (e*ipe) + (i*b), t)

                loss = -(ELBO.mean(0))
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if i % log_every == 0 or i == len(train_loader)-1:
                    print("Epoch: {:02d}/{:02d}, Batch: {:03d}/{:d}, Loss {:9.4f}".format(
                        e, epochs, i, len(train_loader)-1, loss.item()))


    def ELBO(self, logpy: Tensor, logpz: Tensor, logqz: Tensor, i: int, tot: int
             ) -> Tensor:
        """Compute and return the ELBO.

        You could override this method to, for instance, anneal the temperature
        of the KL term during training.

        Args:
            logpy: log-likelihood term; log p_{theta}(y | z, x)
            logpz: log prior term; log p_{theta}(z | x)
            logqz: log approx posterior term; log q_{phi}(z | y, x)
            i: current iteration
            t: total number of iterstions in training process

        Returns:
            Tensor: the batch of single-datapoint ELBOs
        """
        return logpy + logpz - logqz
