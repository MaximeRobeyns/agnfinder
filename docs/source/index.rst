.. AGN Finder documentation master file

AGN Finder documentation
========================

Welcome to the agnfinder documentation!

For instructions on how to install the project, please see the `installation
<installation.html>`_ page. This page contains information for both users and
developers.

This is a fork of the original AGNFinder project by Mike Walmsey and
collaborators, which aims to use (conditional) generative modelling techniques to
map from photometry to distributions over physical parameters :math:`\theta`.

There are broadly two components;

- The first is the sampling code, where a dataset of (:math:`\theta`,
  photometry) pairs are  created, using `Prospector
  <https://ui.adsabs.harvard.edu/abs/2021ApJS..254...22J/abstract>`_. More
  information about the sampling process is in the `sampling <sampling.html>`_
  page.

- The second focuses on the task of inferring galaxy parameters from photometry,
  using the generated datasets from the previous *sampling* component. For a
  description of the methods used and a guide to writing new models or adapting
  some of the existing ones, please see the `inference <inference.html>`_ page.

.. toctree::
   :maxdepth: 1
   :caption: Contents:
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
