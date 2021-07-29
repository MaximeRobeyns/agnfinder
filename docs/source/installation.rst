.. _installation:

Installation Guide
##################

If you are just interested in using this project (i.e. not developing it), then
follow the instructions in the quickstart section. For development, read on to
the following sections.

Quickstart
----------

Begin by cloning the repository::

    git clone https://github.com/maximerobeyns/agnfinder

I recommend that you install the packages using a virtual environment. If you
use the native python virtual environments, then you can run::

    python -m venv agnvenv

This will create a virtual environment called ``agnvenv`` in your current
directory.

.. warning:: This project uses ``Python>=3.9``.

    This is because of features released in this version relating to union
    operators on dictionaries and writing type hints, which we make extensive
    use of.

    You may not have Python 3.9 (or later) available on the machine you intend
    to run ``agnfinder`` on---particularly if you are running on a HPC cluster.

    See the `Building Python3.9`_ section for help.


Installing Development Dependencies
-----------------------------------

FSPS
~~~~

pyFSPS is used to generate the photometric measurements.

Writing Documentation
---------------------

The documentation for this project is written in `sphinx
<https://www.sphinx-doc.org/en/master/>`_, inside a Docker container.

To write documentation, begin by ensuring that you have docker installed, and
that you have the ``agnfinderdocs`` Docker image built locally.

You can either install docker manually by following the `instructions
<https://docs.docker.com/get-docker/>`_ on their site, or by running the
following::

    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh

You can now build the ``agnfinderdocs`` image by running::

    cd ./docs && make img

To use this docker image, run::

    ./docs/writedocs.sh

which will start watching the soruce files in ``./docs/source`` for changes,
compiling the HTML documentation and serving it on ``http://localhost:8081/``.


Building Python3.9
------------------

This is an optional step if you do not have Python 3.9 available on the system
you intend to run ``agnfinder`` on. Here we will assume that you do not have
root privileges.

First, download a Python>=3.9 source code release in some convenient directory.
You could choose to work in ``/tmp``, or any other directory (ideally on your
target machine / architecture). At the time of writing, the latest release can
be downloaded with::

    wget https://www.python.org/ftp/python/3.9.6/Python-3.9.6.tgz

Extract this and go into the source directory::

    tar xzf Python-3.9.6.tgz
    cd Python-3.9.6

We now follow a fairly standard ``./configure && make && make install`` build
procedure. Since we assume that we don't have root privileges, we will
explicitly specify the desired installation prefix during the configuration
stage, as well as providing some other python-specific options::

    ./configure --enable-optimizations --with-ensurepip=install --prefix=$HOME

If you wish to install to another prefix (for instance, you don't want the
resulting executables on some NFS), then replace ``$HOME`` with an appropriate
alternative for your system.

Building and installing is now straightforward::

    make -j<nprocs>
    make install

where ``<nprocs>`` is the number of processes that you are happy to run
concurrently. If compiling on a login node, remember be mindful of other users!

