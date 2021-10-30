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
"""Tests for the MADE (masked autoencoder for distribution estimation)
implementation.

Mostly to verify the autoregressive property is maintained.
"""

import torch as t

from torch.autograd import Variable

import agnfinder.inference.made as made

from agnfinder.inference import CMADE

class MADEParams(made.MADEParams):

    epochs = 1
    batch_size = 128
    dtype = t.float32
    cond_dim = -1
    data_dim = -1
    hidden_sizes = []
    likelihood = made.Gaussian
    likelihood_kwargs = {'data_dim': 9}
    num_masks = -1
    samples = 1
    natural_ordering = False

    def __init__(self, cond_dim, data_dim, hidden_sizes, num_masks, natural_ordering):
        super().__init__()
        self.cond_dim = cond_dim
        self.data_dim = data_dim
        self.hidden_sizes = hidden_sizes
        self.num_masks = num_masks
        self.natural_ordering = natural_ordering

def test_MADE():

    dtype = t.float32
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    D = 9
    configs = [
        (0, D, [200, 200], 128, False),  # no conditioning info
        (8, D, [200, 200], 1, True),
        (8, D, [128, 128], 128, False),
        (8, D, [200, 300], 128, True),
        (8, D, [200, 300], 128, False),
        (8, D, [200, 300, 400], 128, False),
        (8, D, [200, 300, 400], 128, True)
    ]

    for cond_dim, data_dim, hidden_sizes, num_masks, natural_ordering in configs:

        mp = MADEParams(cond_dim, data_dim, hidden_sizes, num_masks,
                        natural_ordering)

        # 50 emulates mini-batch
        x = t.rand((50, data_dim + cond_dim), dtype=dtype, device=device)

        model = CMADE(mp)

        out_size = model.likelihood.n_params() * data_dim

        if device == t.device('cuda'):
            model.cuda()

        # To test the autoregressive property, we use PyTorch's computational
        # graph and run backpropagation for each dimension, which will reveal
        # which other dimensions it depends on.
        #
        # Note that this 'trick' only works with single-ensemble masks.
        midxs = t.randperm(model.num_masks)[:1]

        res = []
        for k in range(out_size):
            xtr = Variable(x, requires_grad=True)
            xtrhat = model(xtr, midxs)
            # select the kth element from the 1st row of outputs as the 'loss'
            loss = xtrhat[0, k]
            loss.backward()

            depends = (xtr.grad[0] != 0).to(dtype=t.uint8)
            depends_ix = list(t.where(depends)[0].cpu().numpy())

            isok = cond_dim + (k % data_dim) not in depends_ix

            assert isok

            res.append((len(depends_ix), k, depends_ix, isok))

        # Useful for debugging:
        res.sort()

        # print("\n")
        # for nl, k, ix, isok in res:
        #     print("Output %2d depends on %2d inputs: %s : %s" %
        #           (k, nl, ix, "PASS" if isok else "NOTOK"))
