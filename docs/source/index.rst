.. AGN Finder documentation master file

AGN Finder documentation
========================

Welcome to the agnfinder documentation!

For instructions on how to install the project, please see the `installation
<installation.html>`_ page. This page contains information for both users and
developers.

This fork is a re-start of the original project by Mike Walmsey and
collaborators, and aims to use (conditional) generative modelling techniques to
map from photometry to distributions over physical parameters :math:`\theta`.

The machine learning components are implemented using PyTorch (as opposed to
TensorFlow, as was the case in the original project). To facilitate this,
methods to create a dataset of :math:`(\theta, \text{photometry})` pairs have
been refactored into a ``sampling`` module. PyTorch ``Tensors`` are used as
opposed to numpy ``ndarrays`` arrays where possible for convenience at the
inference stage. See the `sampling <sampling.html>`_ page for more information about
this module.

Inference happens in one of the classes extending the ``inference`` base class.

..
    For some general tips on getting started and the development workflow, please
    see the `quickstart <./quickstart.html>`_ page.


.. toctree::
   :maxdepth: 1
   :caption: General Topics:
   :numbered:

   installation
   sampling
   inference

..
    Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`
