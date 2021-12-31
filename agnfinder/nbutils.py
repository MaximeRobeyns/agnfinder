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

import math

import corner
import logging
import torch as t
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Any, Sized, Union
from torch.utils.data import DataLoader, Dataset, random_split, Subset

from agnfinder.types import Tensor, column_order
from agnfinder.inference.cvae import CVAE


def plot_corner(samples: Union[np.ndarray, list[np.ndarray]],
                true_params: Optional[list[float]] = None,
                lims: Optional[np.ndarray] = None, labels: list[str] = column_order,
                title: str = "", description: str = ""):
    """Create a corner plot.

    Args:
        samples: the samples to plot. Expects more rows than columns.
        true_params: (optional) overlay true parameter values in red on plot
        lims: (optional; recommended) provide limits for axes. Either list of
            same length as number of columns in `samples`, containing (lower,
            upper) tuples, or list of floating point value for % of points to
            include (see `corner` docs for better description).
        title: the plot title
        description: a descriptive sentence e.g. outlining the training
            procedure for the plot samples

    Note:
        We assume that the samples are in the same order as column_order.
    """

    # silence all non-error logs:
    log = logging.getLogger()
    l = log.getEffectiveLevel()
    log.setLevel(logging.ERROR)

    kwargs: dict[str, Any] = {}
    if true_params is not None:
        kwargs['truths'] = true_params
        kwargs['truth_color'] = '#F5274D'
    kwargs['labels'] = labels
    kwargs['label_kwargs'] = {'fontsize': 12, 'fontweight': 'normal'}

    colours = ["#025159", "#F28705", "#03A696", "#F25D27", "#F20505"]

    # D = 16
    # kwargs['fig'] = plt.figure(figsize=(D,D), dpi=300)
    if lims is not None:
        kwargs['range'] = lims

    if isinstance(samples, list):
        kwargs['color'] = colours[0]
        fig = corner.corner(samples[0], **kwargs)
        for i in range(1, len(samples)):
            kwargs['color'] = colours[i]
            corner.corner(samples[i], fig=fig, **kwargs)
    else:
        kwargs['color'] = '#0A1929'
        fig = corner.corner(samples, **kwargs)

    fig.text(0.05, 1.03, s=title, fontfamily='sans-serif',
             fontweight='demibold', fontsize=25)
    fig.text(0.05, 1.005, s=description, fontfamily='sans-serif',
             fontweight='normal', fontsize=14)
    fig.patch.set_facecolor('white')

    log.setLevel(l)


def plot_posteriors(posterior_ys: np.ndarray, true_ys: np.ndarray,
                    labels: list[str] = column_order, title: str = "",
                    description: str = ""):
    """Plots posterior values p(y | x) on the y axis against the true y values
    on the x axis.

    This allows us to compare parameter estimates over a large number of
    observations.

    Implementation checklist:

    - title
    - descriptive subtitle
    - sub-plot labels and titles
    - nice colours

    Args:
        posterior_theta: samples / expected value / mode from posterior p(y | x)
        true_theta: true galaxy parameters
    """

    assert posterior_ys.shape[1] == true_ys.shape[1]

    base_size = 4

    # number of subplots
    N = true_ys.shape[1]
    cols = min(N, 3)
    rows = math.ceil(N/cols)

    fig, ax = plt.subplots(rows, cols, figsize=(base_size*rows, base_size*cols))
    pltrange = [[0.,1.],[0.,1.]]

    for r in range(rows):
        for c in range(cols):
            i = r * rows + c

            tmp_ax = ax[r][c]

            # ax.hist2d(true_ys[:,i], posterior_ys[:,i], bins=50, range=pltrange, cmap="magma")
            tmp_ax.hist2d(true_ys[:,i], posterior_ys[:,i], bins=50, range=pltrange, cmap="Blues")
            tmp_ax.set_xlabel('True Value')
            tmp_ax.set_ylabel('Posterior Samples')
            tmp_ax.set_title(labels[i])

    fig.text(0.05, 1.05, s=title, fontfamily='sans-serif',
             fontweight='demibold', fontsize=25)
    fig.text(0.05, 1.005, s=description, fontfamily='sans-serif',
             fontweight='normal', fontsize=14)

    fig.tight_layout()


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


def new_sample(dloader: DataLoader, n: int = 1) -> tuple[Tensor, Tensor]:
    dset: Dataset = dloader.dataset
    assert(isinstance(dset, Sized))
    rand_idxs = t.randperm(len(dset))[:n]
    logging.debug('Random test index :', rand_idxs)
    # [n, data_dim]; concatenate along rows: dim 0
    xs, ys = [], []
    for i in rand_idxs:
        tmp_xs, tmp_ys = dset.__getitem__(i)
        xs.append(tmp_xs.unsqueeze(0))
        ys.append(tmp_ys.unsqueeze(0))
    return t.cat(xs, 0).squeeze(), t.cat(ys, 0).squeeze()


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
