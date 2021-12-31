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
"""Tests for the SAN (sequential autoregressive network).
"""

import torch as t
import torch.nn as nn
import numpy as np

from typing import Type, Any, Optional
from torchvision import transforms

import agnfinder.inference.inference as inf
import agnfinder.inference.utils as utils

from agnfinder import nbutils as nbu
from agnfinder.types import Filters, FilterSet
from agnfinder.inference import san


class InferenceParams(inf.InferenceParams):
    model: inf.model_t = san.SAN
    logging_frequency: int = 10000
    filters: FilterSet = Filters.DES
    dataset_loc: str = './data/testdata/photometry_simulation_1000n_z_0p0000_to_4p0000.hdf5'
    retrain_model: bool = True
    use_existing_checkpoints: bool = False
    overwrite_results: bool = True
    ident: str = 'test'


class SANParams(san.SANParams):
    epochs: int = 20
    batch_size: int = 128
    dtype: t.dtype = t.float32

    # Use get_colours_length(cond_dim) if using colours.
    # cond_dim: int = InferenceParams().filters.dim
    cond_dim: int = 8  # size of Euclid filters used to generate testdata

    data_dim: int = 9  # dimensions of data of interest (e.g. physical params)
    module_shape: list[int] = [16, 32]  # shape of the network 'modules'
    sequence_features: int = 4  # features passed between sequential blocks
    likelihood: Type[san.SAN_Likelihood] = san.MoG
    likelihood_kwargs: Optional[dict[str, Any]] = {
            'K': 4, 'mult_eps': 1e-4, 'abs_eps': 1e-4}
    batch_norm: bool = True  # use batch normalisation in network?


# run training and subsequently inference on some test data, ensuring that
# the dimensions are correct at each step.

def test_san_initialisation():
    sp = SANParams()
    model = san.SAN(sp)
    assert model.module_shape == [16, 32]
    assert model.sequence_features == sp.sequence_features


def test_san_likelihood():
    dtype = t.float32
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    ip = InferenceParams()
    sp = SANParams()

    train_loader, test_loader = utils.load_simulated_data(
            path=ip.dataset_loc, split_ratio=ip.split_ratio, batch_size=sp.batch_size,
            transforms=[transforms.ToTensor()], x_transforms=[lambda x: np.log(x)])
    model = san.SAN(sp)

    assert isinstance(model.likelihood, san.MoG)
    assert model.likelihood.name == "MoG"
    # 4 mixture components per output dimension
    assert model.likelihood.K == 4
    # number of parameters required for each dimension of the output
    # (loc, scale, mixture_weight) * number of mixture components
    assert model.likelihood.n_params() == 12
    assert model.last_params is None

    # TODO test _extract_params in likelihood
    # TODO draw test samples and verify dimensions
    # TODO evaluate point likelihoods


def test_san_sequential_blocks():
    model = san.SAN(SANParams())
    test_block: nn.Module
    test_heads: nn.ModuleList

    # 0th block has no additional conditioning parameters
    test_block, test_heads = model._sequential_block(
            cond_dim=8, d=0, module_shape=[16, 32], out_shapes=[4, 2],
            out_activations=[nn.ReLU, None])

    sizes = [8, 16, 32]

    for j, (name, m) in enumerate(test_block.named_modules()):
        m: nn.Module
        if j == 0 or j == 7:
            continue
        i = j-1
        if i % 3 == 0:
            # Linear
            assert name == f'B0L{i//3}'
            assert m.in_features == sizes[i//3]
            assert m.out_features == sizes[(i//3)+1]
        elif i+1 % 3 == 0:
            assert name == f'B0BN{i//3}'  # Batch norm
        elif i+2 % 3 == 0:
            assert name == f'B0A{i//3}'  # Activation
    assert len(test_heads) == 2
    h1, h2 = test_heads
    assert h1.get_submodule("H0:0H0").in_features == 32
    assert h1.get_submodule("H0:0H0").out_features == 4
    assert h2.get_submodule("H0:1H1").in_features == 32
    assert h2.get_submodule("H0:1H1").out_features == 2

    # n>0th blocks have additional conditioning parameters
    test_block, test_heads = model._sequential_block(
            cond_dim=8, d=1, module_shape=[16, 32], out_shapes=[4, 2],
            out_activations=[nn.ReLU, None])
    sizes = [8 + 4 + 1, 16, 32]
    for j, (name, m) in enumerate(test_block.named_modules()):
        m: nn.Module
        if j == 0 or j == 7:
            continue
        i = j-1
        if i % 3 == 0:
            # Linear
            assert name == f'B1L{i//3}'
            assert m.in_features == sizes[i//3]
            assert m.out_features == sizes[(i//3)+1]
        elif i+1 % 3 == 0:
            assert name == f'B1BN{i//3}'  # Batch norm
        elif i+2 % 3 == 0:
            assert name == f'B1A{i//3}'  # Activation
    assert len(test_heads) == 2
    h1, h2 = test_heads
    assert h1.get_submodule("H1:0H0").in_features == 32
    assert h1.get_submodule("H1:0H0").out_features == 4
    assert h2.get_submodule("H1:1H1").in_features == 32
    assert h2.get_submodule("H1:1H1").out_features == 2

    test_block, test_heads = model._sequential_block(
            cond_dim=8, d=40, module_shape=[16, 32], out_shapes=[4, 2],
            out_activations=[nn.ReLU, None])
    sizes = [8 + 4 + 40, 16, 32]
    for j, (name, m) in enumerate(test_block.named_modules()):
        m: nn.Module
        if j == 0 or j == 7:
            continue
        i = j-1
        if i % 3 == 0:
            # Linear
            assert name == f'B40L{i//3}'
            assert m.in_features == sizes[i//3]
            assert m.out_features == sizes[(i//3)+1]
        elif i+1 % 3 == 0:
            assert name == f'B40BN{i//3}'  # Batch norm
        elif i+2 % 3 == 0:
            assert name == f'B40A{i//3}'  # Activation


def test_san_fpath():
    model = san.SAN(SANParams())
    assert model.fpath() == './results/sanmodels/lMoG_cd8_dd9_ms16_32_4_lp12_bnTrue_lr0.0001_e20_bs128.pt'


def test_san_forward():
    """Tests that all tensor shapes and parameter values are as we expect
    during a forward pass."""

    dtype = t.float32
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    ip = InferenceParams()
    sp = SANParams()

    _, test_loader = utils.load_simulated_data(
            path=ip.dataset_loc, split_ratio=ip.split_ratio,
            batch_size=sp.batch_size, transforms=[transforms.ToTensor()],
            x_transforms=[lambda x: np.log(x)])

    model = san.SAN(sp)

    # multiple samples --------------------------------------------------------

    N = 100
    xs, true_ys = nbu.new_sample(test_loader, N)  # photometry, params
    assert xs.shape == (N, sp.cond_dim)

    xs, true_ys = model.preprocess(xs, true_ys)
    assert isinstance(xs, t.Tensor) and isinstance(true_ys, t.Tensor)
    assert xs.shape == (N, sp.cond_dim)
    assert true_ys.shape == (N, sp.data_dim)

    # Explicitly step through the lines in the `forward()` function.
    B = xs.size(0)
    assert B == N
    ys = t.empty((B, 0), dtype=dtype, device=device)
    # last params are used to evaluate a likelihood after a forward pass
    # TODO perhaps try (sp.data_dim, B, n_params) shaped last_params
    last_params = t.ones((B, sp.data_dim, model.likelihood.n_params()),
                         dtype=dtype, device=device)
    assert last_params.shape == (N, 9, 12)

    seq_features = t.empty((B, 0), dtype=dtype, device=device)

    for d in range(model.data_dim):
        d_input = t.cat((xs, seq_features, ys), 1)
        if d == 0:
            assert d_input.shape == (N, 8)
        elif d > 0:
            assert d_input.shape == (N, 8 + 4 + d)

        H = model.network_blocks[d](d_input)
        assert H.shape == (N, 32)

        seq_features = model.block_heads[d][0](H)
        assert seq_features.shape == (N, 4)

        params = model.block_heads[d][1](H)
        assert params.shape == (N, 12)
        y_d = model.likelihood.sample(params).unsqueeze(1)
        assert y_d.shape == (N, 1)

        ys = t.cat((ys, y_d), -1)
        assert ys.shape == (N, d+1)

        last_params[:, d] = params
        assert (last_params[:, d+1:] == 1.).all()
        assert (last_params[:, :d+1] != 1.).all()  # *highly* unlikely that a param is exactly 1.00000
    assert ys.shape == (N, 9)


def test_san_forward_single():
    """Same as above, but for a single point."""

    dtype = t.float32
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    ip = InferenceParams()
    sp = SANParams()
    # Interestingly, using batch norm
    sp.batch_norm = False

    _, test_loader = utils.load_simulated_data(
            path=ip.dataset_loc, split_ratio=ip.split_ratio,
            batch_size=sp.batch_size, transforms=[transforms.ToTensor()],
            x_transforms=[lambda x: np.log(x)])

    model = san.SAN(sp)

    xs, true_ys = nbu.new_sample(test_loader, 1)  # photometry, params
    assert xs.shape == (sp.cond_dim,)
    assert true_ys.shape == (sp.data_dim,)

    xs, true_ys = model.preprocess(xs.unsqueeze(0), true_ys.unsqueeze(0))
    assert isinstance(xs, t.Tensor) and isinstance(true_ys, t.Tensor)
    assert xs.shape == (1, sp.cond_dim)
    assert true_ys.shape == (1, sp.data_dim)

    # Explicitly step through the lines in the `forward()` function.
    B = xs.size(0)
    assert B == 1
    ys = t.empty((B, 0), dtype=dtype, device=device)
    # last params are used to evaluate a likelihood after a forward pass
    # TODO perhaps try (sp.data_dim, B, n_params) shaped last_params
    last_params = t.ones((B, sp.data_dim, model.likelihood.n_params()),
                         dtype=dtype, device=device)
    assert last_params.shape == (1, 9, 12)

    seq_features = t.empty((B, 0), dtype=dtype, device=device)

    for d in range(model.data_dim):
        d_input = t.cat((xs, seq_features, ys), 1)
        if d == 0:
            assert d_input.shape == (1, 8)
        elif d > 0:
            assert d_input.shape == (1, 8 + 4 + d)

        H = model.network_blocks[d](d_input)
        assert H.shape == (1, 32)

        seq_features = model.block_heads[d][0](H)
        assert seq_features.shape == (1, 4)

        params = model.block_heads[d][1](H)
        assert params.shape == (1, 12)
        y_d = model.likelihood.sample(params).unsqueeze(1)
        assert y_d.shape == (1, 1)

        ys = t.cat((ys, y_d), -1)
        assert ys.shape == (1, d+1)

        last_params[:, d] = params
        assert (last_params[:, d+1:] == 1.).all()
        assert (last_params[:, :d+1] != 1.).all()  # *highly* unlikely that a param is exactly 1.00000
    assert ys.shape == (1, 9)
