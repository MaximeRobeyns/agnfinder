import os
import logging
import json  # temp

import numpy as np
import h5py
import tensorflow as tf  # just for eager toggle
os.environ['TF_XLA_FLAGS']='--tf_xla_cpu_global_jit'

from agnfinder.tf_sampling import deep_emulator, api, hmc


def record_performance_on_galaxies(checkpoint_loc, n_galaxies_to_check, n_burnin, n_samples, n_chains, init_method, save_dir):
    emulator = deep_emulator.get_trained_keras_emulator(deep_emulator.tf_model(), checkpoint_loc, new=False)

    _, _, x_test, y_test = deep_emulator.data()
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for i in range(n_galaxies_to_check):
        record_performance(x_test, i, y_test, emulator, n_burnin, n_samples, n_chains, init_method, save_dir)


def record_performance(x_test, i, y_test, emulator, n_burnin, n_samples, n_chains, init_method, save_dir):

    logging.warning('Overriding actual params with fixed test case!')
    # true_params = x_test[i]
    # true_observation = y_test[i]
    with open('data/lfi_test_case.json', 'r') as f:
        test_pair = json.load(f)
        true_params = np.array(test_pair['true_params']).astype(np.float32)
        true_observation = np.array(test_pair['true_observation']).astype(np.float32)

    problem = api.SamplingProblem(true_observation, true_params, forward_model=emulator)
    sampler = hmc.SamplerHMC(problem, n_burnin, n_samples, n_chains, init_method=init_method)
    flat_samples = sampler()

    # explicitly remove old files to avoid shape mismatch issues
    save_file = os.path.join(save_dir, 'galaxy_{}_performance.h5'.format(i))
    if os.path.isfile(save_file):
        os.remove(save_file)

    expected_shape = (n_samples * n_chains, 7)
    if flat_samples.shape != expected_shape:
        logging.warning('Samples not required shape - skipping save to avoid virtual dataset issues')
        logging.warning('actual {} vs expected {}'.format(flat_samples.shape, expected_shape))
    else:
        
        f = h5py.File(save_file, mode='w')
        logging.warning('shape of samples: {}'.format(flat_samples.shape))
        f.create_dataset('samples', data=flat_samples)
        f.create_dataset('true_params', data=true_params)
        f.create_dataset('true_observations', data=true_observation)


def aggregate_performance(save_dir, n_chains, n_samples):
    logging.debug('Creating virtual dataset')
    performance_files = [os.path.join(save_dir, x) for x in os.listdir(save_dir) if x.endswith('_performance.h5')]
    n_sources = len(performance_files)
    logging.warning('Using source files: {}'.format(performance_files))
    logging.debug('Specifing expected data')
    samples_vl = h5py.VirtualLayout(shape=(n_sources, n_chains * n_samples, 7), dtype='f')
    true_params_vl = h5py.VirtualLayout(shape=(n_sources, 7), dtype='f')
    true_observations_vl = h5py.VirtualLayout(shape=(n_sources, 12), dtype='f')

    logging.debug('Specifying sources')
    for i, file_loc in enumerate(performance_files):
        assert os.path.isfile(file_loc)
        samples_source_shape = (n_chains * n_samples, 7)
        logging.warning('shape of samples expected: {}'.format(samples_source_shape))
        samples_vl[i] = h5py.VirtualSource(file_loc, 'samples', shape=samples_source_shape)
        true_params_vl[i] = h5py.VirtualSource(file_loc, 'true_params', shape=(7,))
        true_observations_vl[i] = h5py.VirtualSource(file_loc, 'true_observations', shape=(12,))

    # Add virtual dataset to output file
    logging.debug('Writing virtual dataset')
    with h5py.File(aggregate_filename(save_dir), 'w') as f:
        f.create_virtual_dataset('samples', samples_vl, fillvalue=0)
        f.create_virtual_dataset('true_params', true_params_vl, fillvalue=0)
        f.create_virtual_dataset('true_observations', true_observations_vl, fillvalue=0)
    

def read_performance(save_dir):
    # if the code hangs while reading, there's a shape mismatch between sources and virtual layout - probably samples.
    file_loc = aggregate_filename(save_dir)
    with h5py.File(file_loc, 'r') as f:
        logging.debug('Reading {}'.format(file_loc))
        logging.debug(list(f.keys()))
        samples = f['samples'][...]
        logging.debug('Samples read')
        true_params = f['true_params'][...]
        true_observations = f['true_observations'][...]
    logging.debug('{} loaded'.format(file_loc))
    return samples, true_params, true_observations


def aggregate_filename(save_dir):
    return os.path.join(save_dir, 'all_virtual.h5')

if __name__ == '__main__':

    tf.enable_eager_execution() 
    
    logging.getLogger().setLevel(logging.WARNING)  # some third party library is mistakenly setting the logging somewhere...

    checkpoint_loc = 'results/checkpoints/weights_only/latest_tf'  # must match saved checkpoint of emulator
    n_galaxies = 2
    n_burnin = 2000
    n_samples = 1000
    n_chains = 32
    init_method = 'random'
    # init_method = 'roughly_correct'
    # init_method = 'optimised'
    save_dir = 'results/recovery/latest_{}_{}_{}'.format(n_galaxies, n_samples * n_chains, init_method)

    record_performance_on_galaxies(checkpoint_loc, n_galaxies, n_burnin, n_samples, n_chains, init_method, save_dir)
    aggregate_performance(save_dir, n_samples, n_chains)
    samples, true_params, true_observations = read_performance(save_dir)
    print(samples.shape, true_params.shape, true_observations.shape)
