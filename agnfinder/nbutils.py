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
"""Notebook utilities

This is a file containing utility functions for use in accompanying Jupyter
Notebooks.
"""

import random
import corner
import logging
import torch as t
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Any, Sized
from torch.utils.data import DataLoader, Dataset, random_split, Subset

from agnfinder.types import Tensor
from agnfinder.inference.base import CVAE


def plot_corner(samples: np.ndarray, true_params: Optional[list[float]] = None,
                lims: Optional[np.ndarray] = None):
    """Create a corner plot.

    Args:
        samples: the samples to plot. Expects more rows than columns.
        true_params: (optional) overlay true parameter values in red on plot
        lims: (optional; recommended) provide limits for axes. Either list of
            same length as number of columns in `samples`, containing (lower,
            upper) tuples, or list of floating point value for % of points to
            include (see `corner` docs for better description).
    """
    # silence all non-error logs:
    log = logging.getLogger()
    l = log.getEffectiveLevel()
    log.setLevel(logging.ERROR)

    kwargs: dict[str, Any] = {}
    if true_params is not None:
        kwargs['truths'] = true_params
        kwargs['truth_color'] = '#FF0000'
    if lims is not None:
        kwargs['range'] = lims

    fig = corner.corner(samples, **kwargs)

    log.setLevel(l)
    fig


def plot_violin(dists: list[np.ndarray], labels: list[str],
                true: Optional[list[float]] = None,
                showmean: bool = True, showmedians: bool = False,
                title: str = '', xlabel: str = '', ylabel: str = ''):
    assert len(dists) == len(labels)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=200)
    ax.violinplot(dists, showmeans=showmean, showmedians=showmedians)

    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize='small', rotation=-45)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if true is not None:
        assert len(dists) == len(true)
        inds = np.arange(1, len(true) + 1)
        ax.scatter(inds, true, marker='x', color='red', s=30, zorder=3)
    fig


def new_sample(dloader: DataLoader) -> tuple[Tensor, Tensor]:
    dset: Dataset = dloader.dataset
    assert(isinstance(dset, Sized))
    rand_idx = random.randint(0, len(dset)-1)
    logging.debug('Random test index is: ', rand_idx)
    return dset.__getitem__(rand_idx)


def train_test_split(dataset: Dataset, split_ratio: float = 0.9
                    ) -> tuple[Subset, Subset]:
    assert(isinstance(dataset, Sized))
    n_train = int(len(dataset) * split_ratio)
    n_test = len(dataset) - n_train
    train_set, test_set = random_split(dataset, [n_train, n_test])
    return train_set, test_set



# Visualisations ==============================================================


class LogELBO():
    """
    LogELBO will log the ELBO during training (along with the ELBO
    components).
    """
    def __init__(self, fig, disp, test_loader):
        self.fig = fig
        self.disp = disp
        self.batch_size = 256
        self.tl = test_loader.dataset
        self.ax = plt.subplots()
        self.evals: list[float] = []
        self.LL: list[float] = []
        self.KL: list[float] = []

    def __call__(self, cvae: CVAE):
        cvae.eval()
        self.fig.clear()
        ax = self.fig.add_subplot()
        ax.set_xlabel('Logging Iteration')
        ax.set_ylabel('Test ELBO')
        # ax.set_ylim(-20, 1)
        ELBO_sum: float = 0.
        LL_sum: float = 0.
        KL_sum: float = 0.
        i = 0
        ts = Subset(self.tl, t.randint(len(self.tl), (self.batch_size,)).numpy())
        for x, y in DataLoader(ts, batch_size=self.batch_size):
            try:
                with t.no_grad():
                    x, y = cvae.preprocess(x, y)
                    q = cvae.encoder(y, x)
                    z = q.rsample()
                    pr = cvae.prior(x)
                    p = cvae.decoder(z, x)

                    logpy = p.log_prob(y)  # log p_{theta}(y | z, x)
                    logpz = pr.log_prob(z) # log p_{theta}(z | x)
                    logqz = q.log_prob(z)  # log q_{phi}(z | y, x)
                    # assert logpy.shape == t.Size(int(x.size[0]), 1)

                    LL_sum += float(logpy.mean(0).item())
                    KL_sum += float((logpz - logqz).mean(0).item())

                    ELBO = cvae.ELBO(logpy, logpz, logqz, 0, 0).mean(0).item()
                    ELBO_sum += float(ELBO)
                    i+=1
            except:
                pass
        if i > 0:
            self.evals.append(ELBO_sum / i)
            self.LL.append(LL_sum / i)
            self.KL.append(KL_sum / i)
        ax.plot(np.arange(len(self.evals)), self.evals, label="ELBO")
        ax.plot(np.arange(len(self.LL)), self.LL, label="LL")
        ax.plot(np.arange(len(self.KL)), self.KL, label="KL")
        ax.legend()
        self.fig.tight_layout()
        self.disp.update(self.fig)
        cvae.train()
