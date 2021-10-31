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
"""Entrypoint for CVAE inference tasks."""

import os
import torch as t
import logging

from abc import abstractmethod
from typing import Callable, Optional, Type, Union
from torchvision import transforms
from torch.utils.data import DataLoader

import agnfinder.config as cfg
import agnfinder.inference.cvae_dist as dist

from agnfinder.types import Tensor, DistParams, arch_t
from agnfinder.inference import utils
from agnfinder.inference.inference import Model, ModelParams, InferenceParams
from agnfinder.inference.cvae_base import CVAEPrior, CVAEEnc, CVAEDec, \
                                          _CVAE_Dist, _CVAE_RDist


# Priors ----------------------------------------------------------------------

class StandardGaussianPrior(CVAEPrior):
    """
    A standard Gaussian distribution, whose dimension matches the length of the
    latent code z.
    """

    name: str = "Standard Gaussian"

    def get_dist(self, _: Optional[Union[Tensor, DistParams]]=None) -> _CVAE_Dist:
        mean = t.zeros(self.latent_dim, device=self.device, dtype=self.dtype)
        std = t.ones(self.latent_dim, device=self.device, dtype=self.dtype)
        return dist.Gaussian(mean, std)


class FactorisedGaussianPrior(CVAEPrior):
    """
    Factorised Gaussian prior, whose dimension matches the length of the latent
    code z.

    Example architecture:

    >>> arch_t(layer_sizes=[cond_dim, ...],
    ...        head_sizes=[latent_dim, latent_dim],
    ...        activations=nn.SiLU(),
    ...        [None, Squareplus(1.2)])

    """

    name: str = "Factorised Gaussian"

    def get_dist(self, dist_params: Optional[Union[Tensor, DistParams]] = None
                 ) -> _CVAE_Dist:
        assert dist_params is not None, "Dist params cannot be none"
        assert isinstance(dist_params, list), "Dist params must be a list"
        assert len(dist_params) == 2, "Dist params must contain mean and var"
        [mean, log_std] = dist_params
        std = t.exp(log_std)
        return dist.Gaussian(mean, std)


# Encoders --------------------------------------------------------------------


class FactorisedGaussianEncoder(CVAEEnc):
    """
    A factorised Gaussian encoder; the distribution returned by this encoder
    implements reparametrised sampling and log_prob methods.

    The following is an example of a compatible architecture:

    >>> arch_t(layer_sizes=[data_dim + cond_dim, ...],
    ...        head_sizes=[latent_dim, latent_dim], # loc, scale
    ...        nn.SiLU())

    """

    name: str = "Factorised Gaussian"

    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_RDist:
        assert isinstance(dist_params, list) and len(dist_params) == 2
        [mean, log_std] = dist_params
        std = t.exp(log_std)
        return dist.R_Gaussian(mean, std)


class GaussianEncoder(CVAEEnc):
    """A full-covariance Gaussian encoder

    Expects `dist_params` to be a list of three Tensors;
    1. `mean` giving the location (vector),
    2. `log_std` a vector which gives the elements on the main diagonal of the
        covariance matrix,
    3. `L` is a matrix which will be masked to a lower-triangular matrix before
        exp(log_std) is added; giving the final covariance matrix.

    Here is an example architecture:

    >>> arch_t(layer_sizes=[data_dim + cond_dim, ...],
    ...        head_sizes=[latent_dim, latent_dim, latent_dim*latent_dim],
    ...        nn.SiLU(), [None, nn.ReLU(), nn.ReLU()])

    """

    name: str = "Gaussian"

    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_RDist:
        assert isinstance(dist_params, list) and len(dist_params) == 3
        [mean, log_std, L] = dist_params
        L = L.reshape((-1, log_std.size(-1), log_std.size(-1)))
        std_diag = t.exp(log_std).diag_embed()
        assert std_diag.shape == L.shape
        L = t.tril(L, -1) + std_diag
        return dist.R_MVN(mean, L)


# Decoders --------------------------------------------------------------------


class FactorisedGaussianDecoder(CVAEDec):
    """
    A factorised Gaussian decoder. This corresponds to using a Gaussian
    likelihood in ML training; or equivalently minimising an MSE loss between
    the target data and the reconstruction.

    Example compatible architecture:

    >>> arch = arch_t(layer_sizes=[latent_dim + cond_dim, ...], # input / hidden
    ...               head_sizes=[data_dim, data_dim], # loc and scale
    ...               activations=nn.SiLU(), # input / hidden layer activations
    ...               [None, Squareplus(1.2)]) # scale must be positive

    """

    name: str = "Factorised Gaussian"

    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_Dist:
        assert isinstance(dist_params, list) and len(dist_params) == 2
        [mean, log_std] = dist_params
        std = t.exp(log_std)
        return dist.Gaussian(mean, std)


class MultinomialDecoder(CVAEDec):
    """
    A multinomial decoder. We shouldn't need this for galaxy data, but it's
    here for tests and notebooks using MNIST.

    Example architecture is:

    >>> arch = arch_t(layer_sizes=[latent_dim + cond_dim, ...],
    ...               head_sizes=[data_dim],  # p parameter for multinomial
    ...               activations=SiLU(),
    ...               head_activations=[nn.Softmax()]) # relative probs

    """

    name: str = "Multinomial"

    def get_dist(self, dist_params: Union[Tensor, DistParams]) -> _CVAE_Dist:
        assert isinstance(dist_params, t.Tensor)
        return dist.Multinomial(dist_params)


class LaplaceDecoder(CVAEDec):
    """
    A Laplace likelihood.

    Example compatible architecture (similar to factorised Gaussian models):

    >>> arch = arch_t(layer_sizes=[latent_dim + cond_dim, ...], # input / hidden
    ...               head_sizes=[data_dim, data_dim], # loc and scale
    ...               activations=nn.SiLU(), # input / hidden layer activations
    ...               [None, Squareplus(1.2)]) # scale must be positive

    """

    name: str = "Laplace"

    def get_dist(self, dist_params: Union[t.Tensor, DistParams]) -> _CVAE_Dist:
        assert isinstance(dist_params, list) and len(dist_params) == 2
        [loc, log_scale] = dist_params
        scale = t.exp(log_scale)
        return dist.Laplace(loc, scale)


# CVAE Description ------------------------------------------------------------
# Rather than putting this in agnfinder.types, I find it neater to put this
# closer to the CVAE definition.

class CVAEParams(ModelParams):
    """Configuration class for CVAE.

    This defines some properties which must be provided, and additionally
    performs some validation on those user-provided values.
    """
    def __init__(self):
        super().__init__()
        ri = self.enc_arch.in_shape
        if ri != self.data_dim + self.cond_dim:
            logging.error((
                f'Input dimensions of encoder network ({ri}) '
                f'must equal data_dim ({self.data_dim}) + '
                f'cond_dim ({self.cond_dim}).'))
            raise ValueError("Incorrect CVAE dimensions (see logs)")

        if self.prior_arch is not None:
            pi = self.prior_arch.in_shape
            if pi != self.cond_dim:
                logging.error((
                    f'Input dimensions of prior network ({pi}) '
                    f'must equal cond_dim ({self.cond_dim})'))
                raise ValueError("Incorrect CVAE dimensions (see logs)")

        gi = self.dec_arch.in_shape
        if gi != self.latent_dim + self.cond_dim:
            logging.error((
                f'Input dimensions of decoder network ({gi}) '
                f'must euqal latent_dim ({self.latent_dim}) + '
                f'cond_dim ({self.cond_dim})'))
            raise ValueError("Incorrect CVAE dimensions (see logs)")

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Length of the latent vector, z"""
        pass

    @property
    @abstractmethod
    def adam_lr(self) -> float:
        """Learning rate to use with Adam optimizer"""
        return 1e-3

    @property
    @abstractmethod
    def prior(self) -> Type[CVAEPrior]:
        """Reference to the prior class to use."""
        pass

    @property
    @abstractmethod
    def prior_arch(self) -> Optional[arch_t]:
        """Architecture of 'prior network' p_{theta_z}(z | x)"""
        return None

    @property
    @abstractmethod
    def encoder(self) -> Type[CVAEEnc]:
        """Reference to the encoder / recognition class to use"""
        pass

    @property
    @abstractmethod
    def enc_arch(self) -> arch_t:
        """Architecture of 'recognition network' q_{phi}(z | y, x)"""
        pass

    @property
    @abstractmethod
    def decoder(self) -> Type[CVAEDec]:
        """Reference to the decoder / generation class to use"""
        pass

    @property
    @abstractmethod
    def dec_arch(self) -> arch_t:
        """Architecture of 'generator network' p_{theta_y}(y | z, x)"""
        pass


# Main CVAE Base Class ========================================================


class CVAE(Model):
    """The main base Conditional VAE class

    You must provide the following distributions in the configuration.
    - q_{phi}(z | y, x)   approximate posterior / encoder
    - p_{theta}(z | x)    prior / encoder
    - p_{theta}(y | z, x) generator / decoder

    You can optionally override the `ELBO` method (for instance to implement KL
    warmup). You can also override the `trainmodel` method to implement a
    non-standard training procedure, as well as the `preprocess` method for
    custom data pre-processing.
    """
    def __init__(self, mp: CVAEParams,
                 logging_callbacks: list[Callable] = [],
                 overwrite_results: bool = True) -> None:
        """Initialise a CVAE

        Args:
            mp: CVAE parameters (usually from config.py)
            logging_callbacks: list of callables accepting this model instance;
                often used for visualisations and debugging.
            overwrite_results: whether to overwrite a model with the same
                filepath (read, same parameters) at the end of training. Default:
                True
        """
        super().__init__(mp, overwrite_results, logging_callbacks)

        self.latent_dim = mp.latent_dim

        self.prior = mp.prior(mp.prior_arch, mp.latent_dim, mp.device, mp.dtype)
        self.encoder = mp.encoder(mp.enc_arch, mp.device, mp.dtype)
        self.decoder = mp.decoder(mp.dec_arch, mp.device, mp.dtype)

        if self.prior.is_module:
            self.prior_opt = t.optim.Adam(self.prior.parameters(), lr=mp.adam_lr)
        self.enc_opt = t.optim.Adam(self.encoder.parameters(), lr=mp.adam_lr)
        self.dec_opt = t.optim.Adam(self.decoder.parameters(), lr=mp.adam_lr)

        # Strange mypy error requires this to be put here although it is
        # perfectly well defined and typed in the super class ¯\_(ツ)_/¯
        self.savepath_cached: str = ""

    name: str = 'CVAE'

    def __repr__(self) -> str:
        return (f'{self.name} with {self.prior.name} prior, {self.encoder.name} '
                f'encoder and {self.decoder.name} decoder, a latent dimension of '
                f'{self.latent_dim} trained for {self.epochs} epochs with '
                f'batches of size {self.batch_size}')

    def fpath(self) -> str:
        """Returns a file path to save the model to, based on parameters."""
        # TODO generate a more unique model name based on network architectures
        # (not sequence)
        if self.savepath_cached == "":
            base = './results/cvaemodels/'
            name = (f'p{self.prior.name}_e{self.encoder.name}_d{self.decoder.name}'
                    f'_ld{self.latent_dim}_e{self.epochs}_bs{self.batch_size}')
            if self.overwrite_results:
                self.savepath_cached = f'{base}{name}.pt'
            else:
                n = 0
                while os.path.exists(f'{base}{name}_{n}.pt'):
                    n+=1
                self.savepath_cached = f'{base}{name}_{n}.pt'
        return self.savepath_cached

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
        if logpy.isnan().any():
            logging.warn('logpy is NaN')
            logpy = logpy.nan_to_num()
        if logpz.isnan().any():
            logging.warn('logpz is NaN')
            logpz = logpz.nan_to_num()
        if logqz.isnan().any():
            logging.warn('logqz is NaN')
            logqz = logqz.nan_to_num()
        return logpy + logpz - logqz

    def trainmodel(self, train_loader: DataLoader, ip: InferenceParams,
                   *args, **kwargs) -> None:
        """Train the conditional VAE model.

        Args:
            train_loader: DataLoader to load the training data.
            ip: The parameters to use for training, defined in
                `config.py:InferenceParams`.
        """
        self.train()

        # calculate some constants to help with annealing KL term (optional)
        b = train_loader.batch_size
        assert isinstance(b, int)
        ipe = len(train_loader) * b  # 'iterations per epoch'
        t = self.epochs * ipe

        for e in range(self.epochs):
            for i, (x, y) in enumerate(train_loader):

                # x photometry, y parameters (theta)
                x, y = self.preprocess(x, y)

                # Get q_{phi}(z | y, x)
                q: _CVAE_RDist = self.encoder(y, x)
                z = q.rsample()

                # Get p_{theta}(z | x)
                pr: _CVAE_Dist = self.prior(x)

                # Get p_{theta}(y | z, x)
                p: _CVAE_Dist = self.decoder(z, x)

                logpy = p.log_prob(y)
                logpz = pr.log_prob(z)
                logqz = q.log_prob(z)

                ELBO = self.ELBO(logpy, logpz, logqz, (e*ipe) + (i*b), t)
                loss = -(ELBO.mean(0))

                if self.prior.is_module:
                    self.prior_opt.zero_grad()
                self.enc_opt.zero_grad()
                self.dec_opt.zero_grad()

                loss.backward()

                if self.prior.is_module:
                    self.prior_opt.step()
                self.enc_opt.step()
                self.dec_opt.step()

                if i % ip.logging_frequency == 0 or i == len(train_loader)-1:
                    # Run through all logging functions
                    [cb(self) for cb in self.logging_callbacks]
                    logging.info(
                        "Epoch: {:02d}/{:02d}, Batch: {:05d}/{:d}, Loss {:9.4f}"
                        .format(e, self.epochs, i, len(train_loader)-1,
                                loss.item()))

    def log_cond(self, y: Tensor, x: Tensor, K: int = 1000) -> Tensor:
        """Evaluates log p_{theta}(y | x)

        This uses a Monte Carlo approximation of the marginal likelihood,
        using K samples.

        Args:
            y: the parameter values to find the (conditional) likelihood of
            x: the conditioning photometry values
            K: the number of MC sapmles to use

        Returns:
            Tensor: p_{theta}(y | x) for the provided ys.
        """
        if not self.is_trained:
            logging.warn("CVAE is not yet trained!")

        if y.shape != x.shape:
            try:
                # attempt to broadcast x to be the same shape as y
                x = x.expand((y.size(0), -1))
                assert y.shape == x.shape
            except:
                raise RuntimeError((
                    f'Cannot call log_cond with y of shape: {y.shape} '
                    f'and x of shape: {x.shape}'
                ))

        self.eval()
        x, y = self.preprocess(x, y)

        with t.inference_mode():
            q: _CVAE_RDist = self.encoder(y, x)
            z = q.rsample(t.Size((K,)))
            logqz = q.log_prob(z, nojoint=True).sum(-1)

            pr: _CVAE_Dist = self.prior(x)
            logpz = pr.log_prob(z, nojoint=True).sum(-1)

            flat_zs = z.flatten(0,1)
            tmp_xs = x.repeat_interleave(K, dim=0)
            tmp_ys = y.repeat_interleave(K, dim=0)
            p: _CVAE_Dist = self.decoder(flat_zs, tmp_xs)
            logpy = p.log_prob(tmp_ys).reshape((K, -1))

            return (logpy + logpz -logqz).mean(0)

    def test_generator(self, test_loader: DataLoader) -> float:
        """Test the generator network, p_{theta}(y | z, x)

        Args:
            test_loader: data loader for test set

        Returns:
            float: test loss
        """
        self.eval()
        loss = t.zeros(1, device=self.device, dtype=self.dtype)
        for x, y in test_loader:
            x, y = self.preprocess(x, y)
            pr: _CVAE_Dist = self.prior(x)
            z = pr.sample(t.Size((x.size(0),)))
            p: _CVAE_Dist = self.decoder(z, x)
            NLL = p.log_prob(y)
            loss -= NLL.mean(0)
        return float(loss / len(test_loader))

    def sample(self, x: Tensor, n_samples = 1000, *args, **kwargs) -> Tensor:
        """A convenience method for drawing (conditional) samples from p(y | x)
        for a single conditioning point.

        Args:
            cond_data: the conditioning data; x
            n_samples: the number of samples to draw

        Returns:
            Tensor: a tensor of shape [n_samples, data_dim]
        """
        # TODO implement this
        raise NotImplementedError


# CVAE itself -----------------------------------------------------------------
# You can create variations on the base CVAE class by extending it here.
# For example:

"""
class MyCVAE(CVAE):

    # overload existing methods
    def trainmodel(self, train_loader: DataLoader, epochs: int = 10,
                   log_every: int = 100) -> None:
        return super().trainmodel(train_loader, epochs=epochs, log_every=log_every)

    # add new methods
    def my_method(self):
        pass
"""

if __name__ == '__main__':

    cfg.configure_logging()

    ip = cfg.InferenceParams()
    cp = cfg.CVAEParams()

    train_loader, test_loader = utils.load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=cp.batch_size,
        normalise_phot=utils.normalise_phot_np,
        transforms=[
            transforms.ToTensor()
        ])
    logging.info('Created data loaders')

    model = CVAE(cp)
    logging.info('Initialised CVAE model')

    # NOTE: uses cached model (if available), and saves to disk after training.
    model.trainmodel(train_loader, ip)
    logging.info('Trained CVAE model')


