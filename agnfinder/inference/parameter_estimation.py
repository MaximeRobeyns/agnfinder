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
"""Estimate parameters for a catalogue of real observations."""

import sys
import h5py
import math
import logging
import torch as t

from tqdm import tqdm
from typing import Type, Any
from multiprocessing import Pool, cpu_count

from agnfinder import config as cfg
from agnfinder.types import Filters, FilterSet, column_order
from agnfinder.prospector.load_photometry import load_catalogue, get_filters
from agnfinder.simulation.utils import denormalise_theta

from agnfinder.inference import san
from agnfinder.inference.inference import InferenceParams, model_t
from agnfinder.inference import utils, SAN, CMADE, CVAE, ModelParams


if __name__ == '__main__':
    """
    Load a trained model, load catalogue, estimate median and mode of parameters, save.
    """
    cfg.configure_logging()

    ip = cfg.InferenceParams()
    mp: ModelParams

    if ip.model == SAN:
        mp = cfg.SANParams()
    elif ip.model == CMADE:
        mp = cfg.MADEParams()
    elif ip.model == CVAE:
        mp = cfg.CVAEParams()
    else:
        raise ValueError(f'Unrecognized model {ip.model}')

    model = ip.model(mp)

    # local configurations for reference
    # class IP(InferenceParams):
    #     model: model_t = SAN
    #     logging_frequency: int = 10000
    #     dataset_loc: str = './data/cubes/des_sample/photometry_simulation_40000000n_z_0p0000_to_6p0000.hdf5'
    #     retrain_model: bool = False
    #     use_existing_checkpoints: bool = True
    #     overwrite_results: bool = True
    #     ident: str = 'unit_norm'

    #     # The catalogue of real observations
    #     catalogue_loc: str = './data/DES_VIDEO_v1.0.1.fits'
    #     filters: FilterSet = Filters.DES  # {Euclid, DES, Reliable, All}


    # class MoGSANParams(san.SANParams):
    #     epochs: int = 20
    #     batch_size: int = 1024
    #     dtype: t.dtype = t.float32

    #     cond_dim: int = 7  # dimensions of conditioning info (e.g. photometry)
    #     data_dim: int = 9  # dimensions of data of interest (e.g. physical params)
    #     module_shape: list[int] = [512, 512]  # shape of the network 'modules'
    #     sequence_features: int = 8  # features passed between sequential blocks
    #     likelihood: Type[san.SAN_Likelihood] = san.MoG
    #     likelihood_kwargs: dict[str, Any] = {'K': 10, 'mult_eps': 1e-4, 'abs_eps': 1e-3}
    #     batch_norm: bool = True  # use batch normalisation in network?


    # ip = IP()
    # sp = MoGSANParams()
    # model = san.SAN(sp)

    savepath: str = model.fpath(ip.ident)
    try:
        logging.info(
            f'Attempting to load {model.name} model from {savepath}')
        model.load_state_dict(t.load(savepath))
        model.is_trained = True
        logging.info('Successfully loaded')
        # unsure if this is necessary, but for good measure...
        model.to(model.device, model.dtype)
        model.eval()
    except ValueError:
        logging.info(
            f'Could not load model at {savepath}. Training...')
        sys.exit()

    # Load the catalogue from file
    catalogue = load_catalogue(ip.catalogue_loc, ip.filters, True)
    fs = get_filters(filter_selection=ip.filters)
    required_cols = [f.maggie_col for f in fs]
    xs_np = utils.normalise_phot_np(catalogue[required_cols].values)
    xs = t.from_numpy(xs_np)

    batch: int = 100  # batch size
    N: int = 10000    # samples per posterior

    medianlist: list[t.Tensor] = []
    modelist: list[t.Tensor] = []

    I = math.ceil(xs.shape[0]/batch)

    def modes(sub_batch: t.Tensor) -> t.Tensor:
        sbatch = sub_batch.shape[0]
        histbins = 1000
        binwidth = 1/histbins
        results = t.empty((sbatch, mp.data_dim), dtype=mp.dtype)
        for b in range(sub_batch.shape[0]):
            for d in range(mp.data_dim):
                hist = t.histc(sub_batch[b, :, d], histbins, min=0., max=1.)
                mode = hist.argmax() * binwidth + binwidth/2.
                results[b, d] = mode
        return results

    workers = cpu_count()
    pool = Pool(workers)
    domain_size = math.ceil(batch / workers)

    batched_xs = xs.split(batch)
    print(len(batched_xs))

    # i: int = 0  # index for checkpointing
    for subset in tqdm(batched_xs):
        subset = subset.to(mp.device, mp.dtype)

        # i = idx * batch
        # subset = xs[i:i+batch].to(mp.device, mp.dtype)
        with t.inference_mode():
            subset, _ = model.preprocess(subset, t.empty(subset.shape))
            tmp_samples = model.forward(subset.repeat_interleave(N, 0))

            # 'reshaped samples'
            r_samples = tmp_samples.reshape(-1, N, mp.data_dim).cpu()

            # compute median
            median = r_samples.median(1)[0]
            medianlist.append(median)
            # assert median.shape == (batch, mp.data_dim)

            work = r_samples.split(math.ceil(r_samples.size(0)/workers))
            mode = t.cat(pool.map(modes, work, 1), 0)
            modelist.append(mode)
            # assert mode.shape == (batch, mp.data_dim)

        # # save checkpoint once every 100 subsets
        # i += 1
        # if i % 100 == 0:
        #     mediantensor = t.cat(medianlist, 0)
        #     modetensor = t.cat(modelist, 0)
        #     values = t.cat((mediantensor, modetensor), 1)
        #     t.save(values, f'./results/params/{ip.ident}_{i}.pt')

    mediantensor = t.cat(medianlist, 0)
    modetensor = t.cat(modelist, 0)
    values = t.cat((mediantensor, modetensor), 1)
    t.save(values, f'./results/params/{ip.ident}.pt')
    pool.close()

    # Save results to HDF5.
    pcols = [f'{c}_median' for c in column_order] + \
            [f'{c}_mode' for c in column_order]

    fp = cfg.FreeParams()
    mp.data_dim
    assert len(fp) == mp.data_dim

    denorm_median = denormalise_theta(values[:, :mp.data_dim], fp)
    denorm_mode = denormalise_theta(values[:, mp.data_dim], fp)
    denorm_params = t.cat((denorm_median, denorm_mode), 1)

    save_path = f'./results/params/{ip.ident}.h5'
    with h5py.File(save_path, 'w') as f:
        grp = f.create_group(f'{ip.filters}')

        phot = grp.create_dataset('photometry', data=catalogue.values)
        phot.attrs['columns'] = list(catalogue.columns)
        phot.attrs['description'] = f'''
    Photometric observations from {ip.catalogue_loc}
        '''

        pars = grp.create_dataset('parameter_estimates', data=denorm_params)
        pars.attrs['columns'] = pcols
        pars.attrs['description'] = f'''
    Mediana nd mode estimates for the galaxy parameters.

    Model parameter configuration used:

    {mp}

    Free parameter configuration used:

    {fp}
        '''
