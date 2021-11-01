.. _installation:

Installation Guide
##################

If you are just interested in using this project (i.e. not developing it), then
follow the instructions in the quickstart section. For development, read on to
the following sections.

Quickstart
----------

1. **Python 3.9**

   Please make sure that you have Python 3.9 or later installed on your system.

   You can check your version by running::

       python --version

   Follow the instructions in the `Building Python 3.9`_ section for help if
   you have an older version.

2. **Standard Installation**

   The standard installation process is::

    git clone https://github.com/maximerobeyns/agnfinder
    cd agnfinder
    make install

   The above will only modify files within the ``agnfinder`` directory (and therefore
   doesn't require root privileges).

   2.1 **Specific Python Path**

      If the standard python executable (i.e. the result of running ``which
      python``) is different from the Python >= 3.9 executable that you want to
      use, then update the ``PYTHON`` variable in the ``Makefile`` to point to
      your desired python path, before running ``make install``::

        # Makefile, around line 28
        PYTHON = /absolute/path/to/your/python3.9

   2.2 **Issues with GPy**

       The GPy package requires the Python3 development libraries to build.
       These can be found under ``python3-dev`` on Debian based Linux distros,
       or ``python3-devel`` for others (e.g. Arch, Fedora, RedHat). If you
       encounter issues with the GPy wheel, use your package manager to install
       the Python3 development libraries.

   If you encounter an error during the ``make install`` step, please see the
   `Manual Installation`_ section for guidance.

3. **Shell Setup**

   Every time you run the project, or want to develop using linters and type
   checkers, you should activate the virtual environment and set some necessary
   environment variables (such as ``SPS_HOME``).

   This will be done automatically for you when running targets through the
   ``Makefile``, but if you're running things directly, then you can easily
   setup your shell by running::

    source setup.sh

   To deactivate the virtual environment, either exit your terminal, or type
   ``deactivate``.

4. **(optional)**

   You can test whether the installation was successful by running::

     make test

Manual Installation
-------------------

You can look at the contents of ``./bin/install_with_venv.sh`` to see what the
installation procedure above does.

In order to install ``python-fsps`` (which the project uses), we need to export
an ``SPS_HOME`` environment variable to point a directory where `FSPS
<https://github.com/cconroy20/fsps>`_'s source code lies. In this case, we will
install it in the ``./deps`` directory::

    export SPS_HOME=$(pwd)/deps/fsps

We then get the ``FSPS`` source code by cloning the repository into the
``./deps`` directory::

    git clone https://github.com/cconroy20/fsps.git ./deps/fsps

We now create virtual environment called ``agnvenv``::

    python3.9 -m venv agnvenv

If you prefer to use conda, or install the dependencies somewhere else (for
instance if the current directory is mounted on an NFS), then you can make this
change here.

We then activate the virtual environment, upgrade pip, and install the
dependencies in ``requirements.txt``::

    source agnvenv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

Note that the ``python`` executable above should be the one in the virtual
environment.

We now need to copy some custom sedpy filters into sedpy's filter directory
(which should be a subdirectory of the ``agnvenv``). To find the location of
this filter directory, drop into a python shell (with the venv activated), and
run::

    >>> import sedpy
    >>> print(sedpy.__file__)
    /path/to/agnfinder/agnvenv/lib/python3.9/site-packages/sedpy/__init__.py

This points us to sedpy's installation directory; we want to copy the filters in
``./filters`` to the ``<sedpy-base>/data/filters/`` directory. That is::

    cp -n ./filters/* \
        /path/to/agnfinder/agnvenv/lib/python3.9/site-packages/sedpy/data/filters/

Now we can install ``agnfinder`` itself, by running the following from the root
of the repository::

    pip install -e .


Writing Documentation
---------------------

The documentation for this project is written in `sphinx
<https://www.sphinx-doc.org/en/master/>`_, inside a Docker container.

To write documentation, begin by ensuring that you have docker installed. You
can either install docker manually by following the `instructions
<https://docs.docker.com/get-docker/>`_ on their site, or by running the
following::

    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh


Now, from the root of the repository, you can type ``make docs``. This will
build the ``agnfinderdocs`` docker image if it cannot be found locally (which
will be the case if you are running this for the first time). It will then begin
watching the source files in ``./docs/souce/*.rst`` for changes and re-compile
the HTML. It will also open your browser window at ``http://localhost:8081``
for previewing the documentation.

To stop the docker image, either run ``Ctrl-C`` inside the terminal window, or
if the process hangs for some reason, you can stop it with::

    docker stop $(docker ps -f name=agnfinderdocs -q)

The documentation is re-build and re-deployed using GitHub Actions on pushes to
``master``.


Building Python 3.9
-------------------

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

