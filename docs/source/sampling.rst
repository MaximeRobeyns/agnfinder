.. _sampling:

Photometry Sampling
###################

This section repeatedly runs a forward model on various physical parameters
(redshift, agn_mass, inclination, tage etc.) to simulate photometric
observations, which we use in the `inference <inference.html>`_ section to train a model to
produce the *reverse* mapping: from photometric observations back to
distributions over physical parameters.

The code for the sampling procedure is in the `agnfinder/simulation` directory.

.. note:: Documentation about the sampling procedure itself is missing. For now,
   you can find information about the sampling process, in the `Prospector
   <https://ui.adsabs.harvard.edu/abs/2021ApJS..254...22J/abstract>`_ paper, or
   consult their `documentation <https://prospect.readthedocs.io/_/downloads/en/latest/pdf/>`_.

Running the Sampling Procedure
------------------------------

Running ``make sim`` will run the simulation code in the ``__main__`` section of
``agnfinder/inference/inference.py``, using the configurations in
``agnfinder/config.py``.

This is a multi-threaded program which runs on CPUs only. We use processes
rather than threads due to Python's GIL. With :math:`P` processes used to
produce :math:`N` mock photometric observations from using parameters drawn
uniformly at random from the parameter space, each process works to produce
:math:`\frac{N}{P}` 'observations', with the parameter space partitioned among
processes along the redshift dimension.

Each process writes its samples to a 'partial' results file before exiting. All
of these partial results are finally combined (and shuffled) into a final hdf5
file, ready for use in training a model. Generating 1M samples will result in
approximatly 125Mb of data. Since this combination step is run in-memory, do
bear the memory requirements in mind if you are simulating large catalogues
(e.g. 100M samples).

Free Parameter Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *free parameters* are the physical parameters that are allowed to vary. The
default configuration is as follows::

    class FreeParams(FreeParameters):
        redshift: tuple[float, float] = (0., 6.)
        log_mass: tuple[float, float] = (8, 12)
        log_agn_mass: tuple[float, float] = (-7, math.log10(15))
        log_agn_torus_mass: tuple[float, float] = (-7, math.log10(15))
        dust2: tuple[float, float] = (0., 2.)
        tage: tuple[float, float] = (0.001, 13.8)
        log_tau: tuple[float, float] = (math.log10(.1), math.log10(30))
        agn_eb_v: tuple[float, float] = (0., 0.5)
        inclination: tuple[float, float] = (0., 90.)

This defines our prior assumptions about the reasonable range of values that
these parameters may take, and hence the (in this case) 9-dimensional hyper-cube
from which parameters are uniformly sampled during the simulation process.

Parameters whose names begin with ``log_`` are exponentiated before use. For
example, above the ``log_agn_mass`` parameter constrains the brightness of the
galaxy, from :math:`10^{-7}` to :math:`15`.


Sampling Parameters
~~~~~~~~~~~~~~~~~~~

These are parameters relating to the sampling or simulation procedure itself::

    class SamplingParams(ConfigClass):
        n_samples: int = 40000000
        concurrency: int = os.cpu_count()
        save_dir: str = './data/cubes/my_simulation_results'
        noise: bool = False  # deprecated; ignore
        filters: FilterSet = Filters.DES  # {Euclid, DES, Reliable, All}
        shuffle: bool = True  # whether to shuffle final samples

The ``n_samples`` determines the number of samples generated across all the
processes. The ``concurrency`` parameter is best set according to how much RAM
you have on a particular host: since we use *processes* and not *threads*, the
entire model is re-created for each worker and must be stored in memory.

The ``filters`` parameter selects between pre-defined sets of filters, defined
in ``agnfinder/prospector/load_photometry``. To use a different set of filters,
you should edit this file, and add a new filter to the ``Filters`` class in
``agnfinder/types.py``.

You can find various other parameters relating to the model used to simulate the
photometric observations in the ``agnfinder/config.py`` file.

