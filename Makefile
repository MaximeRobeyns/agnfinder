# AGNfinder: Detect AGN from photometry in XXL data.
#
# Copyright (C) 2021 Maxime Robeyns <maximerobeyns@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

SHELL := bash
.ONESHELL:
# .SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += -j8

# Ensure Python 3.9 is present ------------------------------------------------

# Specify the path to your Python>=3.9 executable here.
PYTHON = $(shell which python3.9)

PYTHON_VERSION_MIN=3.9
PYTHON_VERSION=$(shell $(PYTHON) -c 'import sys; print("%d.%d"% sys.version_info[0:2])' )
PYTHON_VERSION_OK=$(shell $(PYTHON) -c 'import sys;\
  print(int(float("%d.%d"% sys.version_info[0:2]) >= $(PYTHON_VERSION_MIN)))' )

ifeq ($(PYTHON_VERSION_OK),0)
  $(error "Need python $(PYTHON_VERSION) >= $(PYTHON_VERSION_MIN)")
endif

# Targets ---------------------------------------------------------------------

default: test

sim: ## To run the main sampling / simulation program
	@source setup.sh
	@python agnfinder/simulation/simulation.py

saveparams:
	@source setup.sh
	@python bin/to_hdf.py

params: ## Parameter estimation (median and mode) for real world observations
	@source setup.sh
	@python agnfinder/inference/parameter_estimation.py

inf: ## To run the main inference code
	@source setup.sh
	@python agnfinder/inference/inference.py

made: ## To run the MADE inference code specifically
	@source setup.sh
	@python agnfinder/inference/made.py

san: ## To run the SAN inference code specifically
	@source setup.sh
	@python agnfinder/inference/san.py

mcmc: ## To run the MCMC inference code specifically
	@source setup.sh
	@python agnfinder/inference/mcmc.py

cvae: ## To run the CVAE inference code specifically
	@source setup.sh
	@python agnfinder/inference/cvae.py

mypy: ## To run mypy only (this is usually done with test / alltest)
	@source setup.sh
	@mypy

test: mypy  ## To run the program's fast tests (e.g. to verify an installation)
	@source setup.sh
	@python -m pytest -s tests

alltest: mypy ## To run all the program's tests (including slow running ones)
	@source setup.sh
	@python -m pytest tests --runslow

docs: ## To compile the documentation (requires Docker)
	@./docs/writedocs.sh

docsimg: ## To explicitly build the Docker image for writing documentation.
	@docker build -f ./docs/Dockerfile -t agnfinderdocs ./docs

./data/clumpy_models_201410_tvavg.hdf5:
	wget -LO $@ https://www.clumpy.org/downloads/clumpy_models_201410_tvavg.hdf5

install: ./data/clumpy_models_201410_tvavg.hdf5 ## To install everything (warning: downloads ~1.5Gb of data)
	@./bin/install_with_venv.sh $(PYTHON)

qt: ## To re-create the quasar templates (quasar and torus models)
	@source setup.sh
	@python ./agnfinder/quasar_templates.py

kernel:  ## To setup a Jupyter kernel to run notebooks in AGNFinder virtual env
	@source setup.sh
	python -m ipykernel install --user --name agnvenv \
		--display-name "agnvenv (Python 3.9)"

lab: ## To start a Jupyter Lab server
	@source setup.sh
	jupyter lab --notebook-dir=. --ip=0.0.0.0 # --collaborative --no-browser

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	# @./bin/help.sh

.PHONY: alltest cvae docsimg docs inf kernel lab made mypy qt san sim test

# this will force re-installation, even if the clumpy models are already downloaded.
.PHONY: install
