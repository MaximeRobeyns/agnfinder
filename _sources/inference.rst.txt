.. _inference:

Inference Overview
##################

The task of inferring physical galaxy parameters from photometric observations
can be seen as that of learning a mapping :math:`f : \mathcal{X} \to
\Theta`, where :math:`\mathcal{X}` is the space of
:math:`n`-dimensional photometric observations (corresponding to :math:`n`
filters), and :math:`\Theta` is the space of physical parameters, such
as mass, star formation, E(B-V), AGN disk inclination and so forth.

Since the photometric observations :math:`\mathbf{x}\in\mathcal{X}` alone are
unlikely to be sufficient to constrain the full range of physical parameters
:math:`\theta \in \Theta` that we'd like to infer, we resolve to output
distributions physical parameters, conditioned on the information in the
photometric observations. That is, the mapping :math:`f` is one-to-many, and a
reasonable way to deal with this is to work with distributions over the outputs
:math:`p(\theta \vert \mathbf{x})`

From the simulation section, we generated a dataset of :math:`(\theta,
\mathbf{x})` pairs, :math:`\mathcal{D} = \big\{(\theta_{i},
\mathbf{x}_{i})\big\}_{i=1}^{N}`, giving us a fairly standard supervised machine
learning setup.

We can appeal to the broad machine learning literature which presents many ways
to tackle this problem, for instance using generative models or autoregressive
models. Accordingly, to avoid a clash of notation, we will henceforth denote the
physical galaxy parameters as :math:`\mathbf{y}` (previously :math:`\theta`).
This is to match the machine learning nomenclature of denoting the outputs to be
predicted as :math:`\mathbf{y}`, and the model parameters as :math:`\theta`. The
inputs remain :math:`\mathbf{x}`.


Using the Models
----------------

In line with this program's conventions of using configuration classes rather
than command-line arguments, all the options for running inference can be set in
the ``config.py`` file.

There are two main classes to bear in mind:

General Inference Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models in this program are equipped with a ``trainmodel`` method, which provides
a consistent way to train different models. This method's parameters are
contained in the ``InferenceParams`` class (the base class for this is defined
in ``agnfinder/inference/inference.py``).

.. py:class:: InferenceParams(ConfigClass)

   General parameters for the inference code.

   :param model_t model: The model to use for inference.
   :param int split_ratio: The dataset train/test split ratio.
   :param int logging_frequency: How often (in iterations) to output logs during training.
   :param str dataset_loc: Path to a ``hdf5`` file or directory of ``hdf5`` files.
   :param bool retrain_model: Whether to re-train an identically configured model.
   :param bool overwrite_results: Whether to overwrite results from identical model.

   :Example:

       >>> class InferenceParams(inference.InferenceParams):
       ...     model: model_t = san.SAN
       ...     split_ratio: float = 0.8
       ...     logging_frequency: int = 10000
       ...     dataset_loc: str = './data/cubes/photometry_simulation.hdf5'
       ...     retrain_model: bool = True
       ...     overwrite_results: bool = False

Most argument names along with their corresponding type should be self-explanatory.

The ``dataset_loc`` property should point to the output of a simulation run
(that is, the output of running ``make sim``). Please see the `Photometry
Sampling </sampling.html>`_ section for more information about this.

When a model is initialised, a descriptive name is generated based on its
parameters. If the training method (``trainmodel``, see below) is called on a
model with identical parameters to a previously trained and saved model, and
the ``retrain_model`` argument is set to ``False``, then we attempt to load (the
``state_dict`` of) this previous identical model instead of training the model
immediately. If loading fails for some reason (e.g. the file does not exist),
then training proceeds as normal.

If ``retrain_model == True``, then the ``overwrite_results`` argument specifies
what to do when saving the resulting model---if set to ``True``, then the
previously saved model will be overwritten. If set to ``False``, then a number
is appended to the current model's name to make it unique.

Model Parameters
~~~~~~~~~~~~~~~~

While each model has slightly different requirements for its parameterisation
(for instance, the its architecture must be specified), there are some common
parameters which are shared across all models in the codebase.

To reflect this, model parameters inherit a base :class:`ModelParams` class, which
specifies things such as the datatype, device memory to use and so forth.

.. py:class:: ModelParams(ConfigClass)

   General parameters for the inference code.

   :param int epochs: The number of epochs to train this model for.
   :param int batch_size: The mini-batch size.
   :param torch.dtype dtype: PyTorch data type to use in model.
   :param torch.device device: Device (and device memory) to use.
   :param int cond_dim: Dimension of conditioning data (e.g. photometry)
   :param int data_dim: Dimension of output data (e.g. physical params)

   The following example contains a minimal concrete instance of :class:`ModelParams`
   (real models are likely to require additional parameters).

   :Example:

       >>> class ExampleModelParams(ModelParams):
       ...     epochs: int = 20
       ...     batch_size: int = 1024
       ...     dtype: torch.dtype = torch.float32
       ...     device: torch.device = torch.device("cuda")
       ...     cond_dim: int = 8
       ...     data_dim: int = 9

Since all models are concerned with learning a distribution :math:`p(\mathbf{y} \vert
\mathbf{x})`, for :math:`\mathbf{y} \in \mathbb{R}^{N}` and :math:`\mathbf{x}
\in \mathbb{R}^{M}`, we can reliably set parameters ``data_dim = N`` and
``cond_dim = M`` for all models.

**Aside**:

    At first, putting ``epochs`` in the :class:`ModelParams` (instead of the
    ``InferenceParams``) might seem to commit a 'type error': `surely the training
    duration has more to do with the training procedure than the model itself?` The
    ``batch_size`` parameter might also seem similarly misplaced. Since these
    parameters have a large effect on model performance, I claim that they
    should be treated similarly to architectural parameters, and are therefore
    associated with a model.

    For instance, when we come to load a trained model, we `do` care how
    long it was trained for, therefore it makes more sense to associate this
    parameter with the model itself; treating it as a model parameter rather than
    merely a parameter of the training procedure.

Training the models
~~~~~~~~~~~~~~~~~~~

To train a model, first ensure that you have correctly set the model and
inference parameters. Note that these configuration classes needn't necessarily
be the ones in ``config.py``---you are free to define new configuration classes
anywhere in the code.

You will also need a dataset loaded to train the model on. A utility function
(``utils.load_simulated_data``) is available to help with this.

You can now initialise a model by passing the initialised model parameters to
your model's constructor. Finally the ``trainmodel`` method can be called to
run the training procedure.

Note that models are automatically saved to disk after a training run. If you
are re-training an identical parametrised model, the code will first attempt to
load an existing saved model before falling back to running the training
procedure.

The following is a full example, using the `SAN <san_inference.html>`_ model::

    import agnfinder.nbutils as nbu

    # Configure the logger (defaults to INFO-level logs)
    cfg.configure_logging()

    # Initialise the inference, and model parameters; defined in config.py
    ip = cfg.InferenceParams()
    sp = cfg.SANParams()

    # Get the dataloaders for training and testing
    train_loader, test_loader = utils.load_simulated_data(
        path=ip.dataset_loc,
        split_ratio=ip.split_ratio,
        batch_size=sp.batch_size,
        normalise_phot=utils.normalise_phot_np,
        transforms=[
            transforms.ToTensor()
        ])
    logging.info('Created data loaders')

    # Initialise the model
    model = SAN(sp)
    logging.info('Initialised SAN model')

    # Run the training procedure
    model.trainmodel(train_loader, ip)
    logging.info('Trained SAN model')

    # (Example: use the model for something)
    x, _ = nbu.new_sample(test_loader)
    posterior_samples = model.sample(x, n_samples=1000)
    logging.info('Successfully sampled from model')


Creating New Models
-------------------

To ensure that there are consistent interfaces for all the models (to the
benefit of users), and that common code is not duplicated between models (to the
benefit of developers), all the models implemented in the codebase inherit from
an abstract :class:`Model` class (found in ``agnfinder/inference/inference.py:Model``).

Put simply, to create a new model, inherit the :class:`Model` class and ensure that
you have implemented all the abstract properties and methods.

The following shows the constructor, and abstract methods of the :class:`Model`
class.

.. py:class:: Model(torch.nn.Module, ABC)

   Base model class for AGNFinder

   .. py:method:: __init__(self, mp: ModelParams, overwrite_results: bool = False, logging_callbacks: list[Callable] = [])

        :param ModelParams mp: The model parameters.
        :param bool overwrite_results: Overwrite previous results when saving.
        :param list[Callable[[Model], None]] logging_callbacks: Functions executed when logging.

   .. py:method:: name(self) -> str

        Returns a natural-language name for the model.

   .. py:method:: __repr__(self) -> str

        Give a natural-language description of the model. Do include information
        such as ``self.epochs``, ``self.name`` and ``self.batch_size``, as well
        as other architecture-specific details for your specific model.

   .. py:method:: fpath(self) -> str

        Returns a file path to save the model to, which should be unique for
        every different parametrisation of the model.

   .. py:method:: preprocess(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]

        :param Tensor x: The input (usually photometric observations)
        :param Tensor y: The output (usually physical galaxy parameters)
        :returns: The pre-processed parameters (e.g. cast to a specific data type, re-ordered or placed on a specific device's memory.)


   .. py:method:: trainmodel(self, train_loader: DataLoader, ip: InferenceParams, *args, **kwargs) -> None

        :param DataLoader train_loader: The PyTorch DataLoader containing the training data.
        :param InferenceParams ip: Inference parameters containing details of the training procedure.

        Note that any additional model-specific arguments can also be provided.

        This method has a decorator applied in the superclass (which is
        inherited by all sub-classes) which takes care of saving the trained
        model to disk (using :class:`Model.fpath`), as well as loading up an
        existing model rather than repeating training.

   .. py:method:: sample(self, x: Tensor, n_samples: int = 1000, *args, **kwargs) -> Tensor

        :param Tensor x: The conditioning data, :math:`\mathbf{x}`.
        :param int n_samples: The number of samples to draw from the posterior.

        A convenience method for drawing (conditional) samples from :math:`p(\mathbf{y} \vert
        \mathbf{x})` for a single conditioning point.

        Since different models may require additional parameters to arguments to
        perform the sampling, these can be provided using the ``args`` and
        ``kwargs`` parameters.

        This is the only function pertaining to the actual use of the models
        which is required to be consistent across models. Individual models may
        provide different methods to use them.
