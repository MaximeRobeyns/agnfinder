.. _overview:

Program Configuration
#####################

The program's configuration file in ``agnfinder/config.py`` acts as the
*fountain of truth* for all sub-modules. I encourage you to keep configurations (and
the associated results) in version control.

We take a *configuration-as-code* approach, where you can evaluate functions and
instantiate classes as part of the configuration, which all happens inside
configuration *classes*. There is one configuration class per logical section of
the program. These all extend the ``ConfigClass`` utility class, which allows
you to pretty-print out a configuration for logging or debugging by
simply calling ``print(my-config-class)``.

.. py:class:: InferenceParams(ConfigClass)

   General parameters for the inference code.

   :param int epochs: The number of times to run through the simulated dataset during training.
   :param int batch_size: Number of data points to average over for stochastic gradient updates.
   :param split_ratio: The train/test split ratio.
   :param t.dtype dtype: PyTorch data type to use for all neural networks in CVAE.
   :param t.device device: The device to run the models on.
   :param cvat_t CVAE: The CVAE to use for inference.
   :param str dataset_loc: Location of the data generated as a result of running the sampling code.

   :Example:


        >>> class InferenceParams(ConfigClass):
        ...     epochs: int = 8
        ...     batch_size: int = 32
        ...     split_ratio: float = 0.9
        ...     dtype: t.dtype = t.float64
        ...     device: t.device = t.device("cuda")
        ...     model: cvae_t = CVAE
        ...     dataset_loc: str = './data/cubes/photometry_simulation_....hdf5'


.. py:class:: CVAEParams(ConfigClass, base.CVAEParams)

   A self-contained description of the CVAE to use for inference.

   :param int cond_dim: Dimension of photometry data (i.e. number of filters)
   :param int data_dim: Dimensions of physical parameters to predict
   :param int latent_dim: Dimension of latent space :math:`z`.

   :param CVAEPrior prior: prior :math:`p_{\theta}(z \vert x)`
   :param arch_t prior_arch: architecture to use for prior network.

   :param CVAEEnc encoder: encoder model :math:`q_{\phi}(z \vert y, x)`
   :param arch_t enc_arch: architecture to use for the encoder network

   :param CVAEDec decoder: decoder model :math:`p_{\theta}(y \vert z, x)`
   :param arch_t dec_arch: architecture to use for decoder network

   :Warning: The network architecture used for each of :math:`p_{\theta}(z \vert)x`, :math:`q_{\phi}(z \vert y, x)` and :math:`p_{\theta}(y \vert z, x)` must be compatible with the corresponding distribution. See the doc strings in the ``agnfinder/inference/inference.py`` file for example compatible architectures.

   :Example:

        >>> class CVAEParams(ConfigClass, base.CVAEParams):
        ...     cond_dim = 8
        ...     data_dim = 9  # len(FreeParameters())
        ...     latent_dim = 4
        ...
        ...     prior = inference.StandardGaussianPrior
        ...     prior_arch = None
        ...
        ...     encoder = inference.GaussianEncoder
        ...     enc_arch = arch_t(
        ...         layer_sizes=[data_dim + cond_dim, 32],
        ...         activations=nn.SiLU(),
        ...         head_sizes=[latent_dim, latent_dim, latent_dim**2],
        ...         head_activations=[None, nn.ReLU(), nn.ReLU()],
        ...         batch_norm=True)
        ...
        ...     decoder = inference.FactorisedGaussianDecoder
        ...     dec_arch = arch_t(
        ...         layer_sizes=[latent_dim + cond_dim, 32],
        ...         head_sizes=[data_dim, data_dim],
        ...         activations=nn.SiLU(),
        ...         head_activations=[None, Squareplus(1.2)],
        ...         batch_norm=True)

.. py:class:: FreeParams(ConfigClass)

    The free parameters class specifies the physical galaxy parameters that we
    attempt to recover from photometric observations. Keys prefixed by ``log_`` will
    automatically be exponentiated later. These are all tuples of floats
    (representing the ranges or bounds for these parameters).

    :Example:

        >>> class FreeParameters(ConfigClass):
        ...     redshift: tuple[float, float] = (0., 4.)
        ...     log_mass: tuple[float, float] = (8, 12)
        ...     log_agn_mass: tuple[float, float] = (-7, math.log10(15))
        ...     log_agn_torus_mass: tuple[float, float] = (-7, math.log10(15))
        ...     dust2: tuple[float, float] = (0., 2.)
        ...     tage: tuple[float, float] = (0.001, 13.8)
        ...     log_tau: tuple[float, float] = (math.log10(.1), math.log10(30))
        ...     agn_eb_v: tuple[float, float] = (0., 0.5)
        ...     inclination: tuple[float, float] = (0., 90.)


.. py:class:: SamplingParameters(ConfigClass)

    These are program parameters for the sampling procedure, where the dataset of
    (params, photometry) pairs is simulated.

    :param int n_samples: A value of about 100,000 for this will take about 10 minutes to run on a modestly fast computer. There is little point in running the sampling procedure on a HPC cluster since it is single-threaded.
    :param FilterSet filters: The ``filters`` parameter accepts a ``FilterSet`` value; this enum is defined in ``agnfinder/types.py``, and can either be ``Euclid`` (8 filters), ``Reliable`` (12 filters), or ``All`` (12 filters).


    :Example:

        >>> class SamplingParams(ConfigClass):
        ...     n_samples: int = 100000
        ...     redshift_min: float = 0.
        ...     redshift_max: float = 4.
        ...     save_dir: str = './data/cubes'
        ...     noise: bool = False
        ...     filters: FilterSet = Filters.Euclid

.. py:class:: CPzParams(ConfigClass)

   Configuration for the CPz ("classification-aided photometric-redshift
   estimation") parameters.

   Boolean attributes can be turned on or off as you please.

   Attributes of type ``MaybeFloat`` must either be

       - ``Just(value)``, where ``value`` is a floating point number.
       - ``Free``

   Attributes of type ``Optional`` must either be

       - ``OptionalValue(<MaybeFloat value>)``, or
       - ``Nothing``.

   (These 'monadic' data types are defined in ``agnfinder/types.py`` and give
   better 'type safety' to configuration values.)

   :Example:


        >>> class CPzParams(ConfigClass):
        ...     dust: bool = True
        ...     model_agn: bool = True
        ...     igm_absorbtion: bool = True
        ...
        ...     # Non-optional values {Free | Just(<float>)}
        ...     agn_mass: MaybeFloat = Free
        ...     redshift: MaybeFloat = Free
        ...     inclination: MaybeFloat = Free
        ...     fixed_metallicity: MaybeFloat = Just(0.)  # solar metallicity
        ...
        ...     # Optional values
        ...     # {Nothing | OptionalValue(Free) | OptionalValue(Just(<float>))}
        ...     agn_eb_v: Optional = OptionalValue(Free)
        ...     agn_torus_mass: Optional = OptionalValue(Free)


.. py:class:: SPSParams(ConfigClass)

   Configuration parameters for FSPS stellar population synthesis.

   :Note:

        The ``emulate_ssp`` parameter is deprecated; leave as ``False``.


   :Example:

        >>> class SPSParams(ConfigClass):
        ...     zcontinuous: int = 1
        ...     vactoair_flag: bool = False
        ...     compute_vega_mags: bool = False
        ...     reserved_params: list[str] = ['zred', 'sigma_smooth']
        ...
        ...     emulate_ssp: bool = False
