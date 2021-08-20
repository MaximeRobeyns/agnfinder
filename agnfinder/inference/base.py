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

from typing import Union
from torch.utils.data import DataLoader
from torch.distributions import LowRankMultivariateNormal

from agnfinder.types import Tensor, Distribution, DistParam, arch_t, CVAEParams


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

    def forward(self, x: Tensor) -> Union[Tensor, DistParam]:
        """Forward pass through the network

        Args:
            x: input tensor

        Returns:
            Union[Tensor, DistParam]: Return single tensor if there is a single
                head, otherwise a list of tensors (DistParam) with each element
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


class RecognitionNet(MLP, abc.ABC):
    """An abstract recognition network; returning the *parameters* of

        q_{phi}(z | y, x)

    as a list of tensors.
    """

    def __init__(self, arch: arch_t, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        MLP.__init__(self, arch, device, dtype)
        abc.ABC.__init__(self)

    def __call__(self, y: Tensor, x: Tensor) -> DistParam:
        return self.dist_params(y, x)

    @abc.abstractmethod
    def dist_params(self, y: Tensor, x: Tensor) -> DistParam:
        raise NotImplementedError


class PriorNet(MLP, abc.ABC):
    """Abstract 'prior' network; implementing

        p_{theta_x}(z | x).

    """

    def __init__(self, arch: arch_t, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        MLP.__init__(self, arch, device, dtype)
        abc.ABC.__init__(self)

    def __call__(self, x: Tensor) -> Distribution:
        return self.distribution(x)

    @abc.abstractmethod
    def distribution(self, x: Tensor) -> Distribution:
        raise NotImplementedError


class GeneratorNet(MLP, abc.ABC):
    """Abstract generation network; implementing

        p_{theta_y}(y | z, x).

    """

    def __init__(self, arch: arch_t, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        MLP.__init__(self, arch, device, dtype)
        abc.ABC.__init__(self)

    def __call__(self, z: Tensor, x: Tensor) -> Distribution:
        return self.distribution(z, x)

    @abc.abstractmethod
    def distribution(self, z: Tensor, x: Tensor) -> Distribution:
        raise NotImplementedError


class EKS(object):
    """Entropy Kaos Service haha (y8OnoxKotPQ)

    This is just a standard Gaussian with convenience features; useful for
    sampling epsilon during reparametrisation.
    """
    def __init__(self, latent_shape: int, device: t.device = t.device('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        # super().__init__(t.zeros(latent_shape).to(device, dtype),  # mean
        #                  t.eye(latent_shape).to(device, dtype))  # diag
        self._ls = latent_shape
        self.device = device
        self.dtype = dtype

    def __call__(self, size: Union[t.Size, int]) -> Tensor:
        shape = size if isinstance(size, t.Size) else t.Size((size, self._ls))
        return t.randn(shape).to(self.device, self.dtype)
        # return super().rsample(shape).to(self.device, self.dtype)

    def log_prob(self, x):
        raise NotImplementedError


class CVAE(nn.Module, abc.ABC):
    """Conditional VAE class

    Implementation checklist:

    - recognition_params(self, y: Tensor, x: Tensor) -> DistParam
    - prior(self, x: Tensor) -> Distribution
    - generator(self, z: Tensor, x: Tensor) -> Distribution
    - rsample(self, y: Tensor, x: Tensor) -> tuple[Tensor, DistParam]
    - kl_div(sef, z: Tensor, x: Tensor, rparams: DistParam) -> Tensor
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
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.latent_dim: int = cp.latent_dim

        self.EKS = EKS(cp.latent_dim, device, dtype)

        self.recognition_net = MLP(cp.recognition_arch, device, dtype)
        if cp.prior_arch is not None:
            self.prior_net = MLP(cp.prior_arch, device, dtype)
        self.generator_net = MLP(cp.generator_arch, device, dtype)

        # NOTE should this be in the abstract base class?
        # Making big assumption about using same Adam opt with default params on all nets.
        nets: list[MLP] = [getattr(self, n) for n in dir(self) \
                           if n.endswith('_net') and \
                           isinstance(getattr(self, n), MLP)]
        self.opt = t.optim.Adam([param for n in nets for param in n.parameters()])

    def preprocess(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        return x.to(self.device, self.dtype), y.to(self.device, self.dtype)

    def trainmodel(self, train_loader: DataLoader, epochs: int = 10,
                   log_every: int = 100):
        for e in range(epochs):
            for i, (x, y) in enumerate(train_loader):

                x, y = self.preprocess(x, y)

                z, rparams = self.rsample(y, x)

                KL = self.kl_div(z, x, rparams)

                LL = self.log_likelihood(y, z, x)

                ELBO = LL - KL
                loss = (-ELBO).mean(0)  # average across batches

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if i % log_every == 0 or i == len(train_loader)-1:
                    print("Epoch: {:02d}/{:02d}, Batch: {:03d}/{:d}, Loss {:9.4f}".format(
                        e, epochs, i, len(train_loader)-1, loss.item()))

    def log_likelihood(self, y: Tensor, z: Tensor, x: Tensor) -> Tensor:
        """Evaluate p(y | z, x)

        Evaluate the probability of the given training point y under our
        likelihod, parametrised by the generative network, whose inputs are the
        z and x tensors.

        For a Gaussian likelihood, this is equivalent to outputting the
        negative MSE loss. For a Bernoulli likelihood, this is equivalent to
        outputting the negative cross entropy.

        Args:
            y: batch of training samples
            z: batch of latent variable samples
            x: batch of conditioning information

        Returns:
            Tensor: log likelihood (still batched)
        """
        gen_dist = self.generator(z, x)
        return gen_dist.log_prob(y)

    def evidence(self, y, x):
        """
        Once the model is trained, use this method to evaluate the model
        evidence under the held-out test dataset.
        """
        # TODO implement me!
        raise NotImplementedError

    @abc.abstractmethod
    def recognition_params(self, y: Tensor, x: Tensor) -> DistParam:
        """Returns parameters of q_{phi}(z | y, x) distribution.

        These will be used to obtain reparametrised z samples.

        Args:
            y: training output sample (batch)
            x: training conditioning info sample (batch)

        Returns:
            DistParam: parameters of approx posterior distribution q
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prior(self, x: Tensor) -> Distribution:
        """Returns the prior distribution p_{theta_z}(z | x)

        Should use the prior_net internally to generate the parameters of this
        distribution. Otherwise this function might simply return a standard
        Gaussian; without making use of a prior_net at all.

        Args:
            x: conditioning information (e.g. photometric observations)

        Returns:
            Distribution: prior distribution over latent vectors
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generator(self, z: Tensor, x: Tensor) -> Distribution:
        """Returns the generator distribution p_{theta_y}(y | z, x)

        Should use the generator_net internally to generate the parameters of this
        distribution.

        Args:
            z: (rsampled / arbitrarily provided) latent vector
            x: conditioning information

        Returns:
            Distribution: likelihood / distribution over outputs
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rsample(self, y: Tensor, x: Tensor) -> tuple[Tensor, DistParam]:
        """Reparametrised sampling of latent vector

        Args:
            y: data (batch); e.g. physical galaxy parameters
            x: conditioning information (batch) e.g. photometric observations

        Returns:
            tuple[Tensor, DistParams]:
                1. (Tensor) The latent variable sample, z.
                2. (DistParams) The parameters of the mapping g.
                   This will usually include the epsilon sample, as well as the
                   distribution parameters (e.g. mean and (log) covariance).
                   Please also note that DistParams = list[torch.Tensor].

                   By convention, include epsilon as the 0th element of
                   this list of Tensors, and other parameters thereafter.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def kl_div(self, z: Tensor, x: Tensor, rparams: DistParam) -> Tensor:
        """Evaluate KL[q_{phi}(z | y, x) || p_{theta}(z | x)]

        This will usually involve an application of the change-of-variables
        formula, or an analytic form of the KL divergence (for instance,
        between two Gaussians).

        For the second term in the KL divergence, you will have to evaluate the
        prior likelihood of z (given x); p(z | x).

        Args:
            z: sampled latent variable
            rparams: reparametrised sampling parameters used to generate the
                latent variable sample. By convention, the 0th element of this
                list is the epsilon sample used to generate z.

        Returns:
            Tensor: the (mini-batched) KL divergence
        """
        raise NotImplementedError


# New, hopefully 'better' conditional VAE
class NCVAE(nn.Module, abc.ABC):
    """
    Perhaps a cleaner abstraction would be to break out each of the distributions:
    - q_{phi}(z | y, x)   approximate posterior / encoder
    - p_{theta}(z | x)    prior / encoder
    - p_{theta}(y | z, x) generator / decoder

    And provide methods to 'initialise' the distribution (i.e. get distribution
    parameters given conditioning information), as well as sampling from and
    evaluating the log_prob of a datapoint under this distribution.

    We would ideally like to keep these encoder, prior and decoder networks
    modular, so that we can swap out just one of them and re-use others if need
    be.

    Also need to think more about how configuration will work (in config.py).
    Do we directly specify the classes to use in the configuration struct, or
    have some other 'class discovery' mechanism?
    """

    def __init__(self, cp: CVAEParams,
                 device: t.device = t.deivce('cpu'),
                 dtype: t.dtype = t.float64) -> None:
        """Initialises a CVAE

        TODO Provide the prior, encoder and decoder classes to be used as arguments.
        """

        def trainmodel(self, train_loader: DataLoader, epochs: int = 10,
                       log_every: int = 100):
            """
            This is a potentially better way of organising the training process
            and encapsulating the functionality of the prior, encoder and
            decoder distributions for better modularity and separation of
            concerns.
            """
            for e in range(epochs):
                for i, (x, y) in enumerate(train_loader):

                    x, y = self.preprocess(x, y)

                    # obtain the parameters of the q distribution (used for r-sampling).
                    q_params = self.decoder.params(y, x)
                    z = self.decoder.sample(q_params)
                    logqz = self.decoder.log_prob(z, q_params)

                    prior_params = self.prior.params(x)
                    logpz = self.prior.log_prob(z, prior_params)

                    p_params = self.decoder.params(z, x)
                    logpy = self.decoder.log_prob(y, p_params)

                    ELBO = logpy + logpz - logpz

                    loss = (-ELBO).mean(0)

                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()













