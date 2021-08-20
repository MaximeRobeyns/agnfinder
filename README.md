<div align="center">
  <img src="https://github.com/MaximeRobeyns/agnfinder/raw/master/docs/source/_static/base_logo.png" height="150px">
</div>

[![Compile docs](https://github.com/MaximeRobeyns/agnfinder/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/MaximeRobeyns/agnfinder/actions/workflows/docs.yml)

Detect AGN from photometry in XXL data, as Euclid prep.

This fork explores the use of conditional generative modelling to to estimate galaxy parameters, using PyTorch.

Please see the [documentation](https://maximerobeyns.github.io/agnfinder/) for more information about this project; in particular, the description of the [inference procedure using CVAEs](https://maximerobeyns.github.io/agnfinder/inference.html) may give a good overview of this project's goals.

## Installation

The [installation guide](https://maximerobeyns.github.io/agnfinder/installation.html) contains detailed information on how to install the project, but for users looking to get started quickly, the following steps should be sufficient.

AGNFinder only supports **Python 3.9** onwards, since it makes use of recent language features (as of August, 2021).

To install, run
```
git clone https://github.com/MaximeRobeyns/agnfinder
cd agnfinder
make install
```

Please note that this will download some necessary data (approx ~1.5Gb), and as such may take a while on slower internet connections. It will only modify files within the ``/agnfinder`` directory, and can be run without root privileges.

To make sure your installation was successful, run
```
make test
```

For further help and available commands, run
```
make help
```
or consult the [documentation](https://maximerobeyns.github.io/agnfinder/).

## Licence

[GPL-3.0 License](LICENSE)

