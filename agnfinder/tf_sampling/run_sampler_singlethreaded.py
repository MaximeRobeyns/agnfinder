import argparse
import logging
import os

from tqdm import tqdm
import numpy as np
import tensorflow as tf  # just for eager toggle

from agnfinder.tf_sampling import run_sampler, deep_emulator

# TODO will change to some kind of unique id for each galaxy, rather than the index
def get_galaxies_without_results(n_galaxies):
    without_results = []
    i = 0
    while len(without_results) < n_galaxies:
        if not os.path.isfile(run_sampler.get_galaxy_save_file(i, save_dir)):
            without_results.append(i)
        i += 1
    return without_results

def record_performance_on_galaxies(checkpoint_loc, max_galaxies, n_burnin, n_samples, n_chains, init_method, save_dir):
    emulator = deep_emulator.get_trained_keras_emulator(deep_emulator.tf_model(), checkpoint_loc, new=False)

    _, _, x_test, y_test = deep_emulator.data()
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    n_batches = 1  # TODO hardcode for now
    for _ in tqdm(range(n_batches)):
        galaxy_indices = get_galaxies_without_results(n_chains)  # commenting out for now
        # logging.critical('For now, only running on this specific galaxy!')
        # assert n_chains == 1
        # galaxy_indices = [1977]  # galaxy in 10m param cube w/ all params close to 0.5

        true_params = x_test[galaxy_indices]  # true params etc. now have a batch dimension
        true_observation = deep_emulator.denormalise_photometry(y_test[galaxy_indices])
        run_sampler.sample_galaxy_batch(galaxy_indices, true_observation, true_params, emulator, n_burnin, n_samples, n_chains, init_method, save_dir)


if __name__ == '__main__':

    """
    Run the emulated HMC method on many galaxies, in a single thread.
    Evaluating performance at recovering posteriors can be done in `evaluate_performance.py`

    Example use: 
    python agnfinder/tf_sampling/run_sampler_singlethreaded.py --checkpoint-loc results/checkpoints/latest --output-dir results/emulated_sampling --n-chains 4 --n-samples 100 --n-burnin 100 --init random
    python agnfinder/tf_sampling/run_sampler_singlethreaded.py --checkpoint-loc results/checkpoints/latest --output-dir results/emulated_sampling

    """
    parser = argparse.ArgumentParser(description='Run emulated HMC on many galaxies')
    parser.add_argument('--checkpoint-loc', type=str, dest='checkpoint_loc')
    parser.add_argument('--output-dir', dest='output_dir', type=str)  # in which save_dir will be created
    parser.add_argument('--max-galaxies', type=int, default=1, dest='max_galaxies')
    parser.add_argument('--n-burnin', type=int, default=1000, dest='n_burnin')  # below 1000, may not find good step size
    parser.add_argument('--n-samples', type=int, default=6000, dest='n_samples')  # 6000 works well?
    parser.add_argument('--n-chains', type=int, default=96, dest='n_chains')  # 96 is ideal on my laptop, more memory = more chains free
    parser.add_argument('--init', type=str, dest='init_method', default='optimised', help='Can be one of: random, roughly_correct, optimised')
    args = parser.parse_args()

    tf.enable_eager_execution() 
    
    logging.getLogger().setLevel(logging.INFO)  # some third party library is mistakenly setting the logging somewhere...

    checkpoint_loc =  args.checkpoint_loc
    output_dir = args.output_dir
    assert checkpoint_loc is not None
    assert output_dir is not None
    max_galaxies = args.max_galaxies
    n_burnin = args.n_burnin
    n_samples = args.n_samples
    n_chains = args.n_chains
    init_method = args.init_method
    save_dir = os.path.join(output_dir, 'latest_{}_{}_{}'.format(n_samples, n_chains, init_method))

    record_performance_on_galaxies(checkpoint_loc, max_galaxies, n_burnin, n_samples, n_chains, init_method, save_dir)
    # run_sampler.aggregate_performance(save_dir, n_samples, chains_per_galaxy=1)
    # samples, true_params, true_observations = run_sampler.read_performance(save_dir)
    # print(samples.shape, true_params.shape, true_observations.shape)
