.. AGN Finder documentation master file

`GitHub Project <https://github.com/MaximeRobeyns/agnfinder>`_

AGN Finder documentation
========================

Welcome to the agnfinder documentation!

-------------------------------------------------------------------------------

If you're looking for installation instructions, please see the `installation
<installation.html>`_ page.

-------------------------------------------------------------------------------

This is a fork of the original AGNFinder project by Mike Walmsey and
collaborators, which aims to use (conditional) generative modelling techniques to
map from photometry to distributions over physical parameters :math:`\theta`.

There are broadly two components:

1. **Simulation and Dataset Creation**

   The first section is for *simulation*, where you can create a dataset
   of simulated :math:`(\text{galaxy parmeter}, \text{photometry})` pairs.
   This uses `Prospector
   <https://ui.adsabs.harvard.edu/abs/2021ApJS..254...22J/abstract>`_ behind the
   scenes, and should be the first step if you are installing and running this
   repository for the first time.

   See the `Photometry Sampling </sampling.html>`_ section for more details about
   this stage.

2. **Inference**

   In this second section of the codebase we deal with the task of estimating
   the distribution over physical galaxy parameters, which we denote
   :math:`\mathbf{y}`, given photometric observations :math:`\mathbf{x}`; that
   is, estimating
   :math:`p(\mathbf{y} \vert \mathbf{x})`.

   This is a standard supervised machine learning setup. There are several
   candidate models in the codebase, and the code has been structured such that
   it is easy to implement new models, and compare them to the existing methods.

   To see how to do this, and for an overview of the existing models, please
   read the `Inference </inference.html>`_ page.

Configurations
--------------

In this project, we take a *configuration-as-code* approach, where the program's
configuration is in ``agnfinder/config.py``.

The main reasons for doing it this way are that:

- configurations are all located in one place instead of being scattered around
  the code
- the program arguments and parameters are maintained in version control, as an
  example for new users and a reference for old users
- there are no long command line arguments to type by hand, and configurations
  don't litter submission scripts and ``Makefiles``.
- you can evaluate arbitrary code to set configuration values. (In general
  this is a security risk, but not for this application.)

This is by no means perfect, and many folks prefer to do things differently. If
this is the case, you are welcome to modify your ``config.py`` to, for instance,
call ``argparse`` and accept command line arguments.


Miscellaneous
-------------

We use `type hints <https://www.python.org/dev/peps/pep-0484/>`_ throughout the
code to allow for static type checking using `mypy <http://mypy-lang.org/>`_. In
general this helps to catch bugs, and makes it clearer to folks unfamiliar with
the code what an object is, or what a function is doing.

Tests are written using `pytest <https://pytest.org>`_.

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   :numbered:

   installation
   basic_usage
   sampling
   inference
   san_inference
   cvae_inference

..
    Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`
