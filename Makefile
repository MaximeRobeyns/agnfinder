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
.SHELLFLAGS := -eu -o pipefail -c
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

# Generic 'run' target for development
sim: ## To run the main sampling / simulation program
ifndef SPS_HOME
		@source setup.sh
endif
	@python agnfinder/simulation/simulation.py

# 'inference': train CVAE
inf: ## To run the inference code
ifndef SPS_HOME
		@source setup.sh
endif
	@python agnfinder/inference/inference.py

mypy: ## To run mypy only (this is usually done with test / alltest)
ifndef SPS_HOME
		@source setup.sh
endif
	@mypy

test: mypy  ## To run the program's fast tests (e.g. to verify an installation)
ifndef SPS_HOME
		@source setup.sh
endif
	@python -m pytest -s tests

alltest: mypy ## To run all the program's tests (including slow running ones)
ifndef SPS_HOME
		@source setup.sh
endif
	@python -m pytest tests --runslow

docs: ## To compile the documentation (requires Docker)
	@./docs/writedocs.sh

docsimg: ## To explicitly build the documentation writing image
	@docker build -f ./docs/Dockerfile -t agnfinderdocs ./docs

./data/clumpy_models_201410_tvavg.hdf5:
	wget -O $@ https://www.clumpy.org/downloads/clumpy_models_201410_tvavg.hdf5

# Install agnfinder project locally
install: ./data/clumpy_models_201410_tvavg.hdf5 ## To install everything (warning, downloads ~1.5Gb of data)
	@./bin/install_with_venv.sh $(PYTHON)

# Runs the main entrypoint for visualising the quasar templates
qt: ## To re-create the quasar templates (quasar and torus models)
ifndef SPS_HOME
		@source setup.sh
endif
	@python ./agnfinder/quasar_templates.py

kernel:  ## To setup a Jupyter kernel to run notebooks in AGNFinder virtual env
ifndef SPS_HOME
		@source setup.sh
endif
	python -m ipykernel install --user --name agnvenv \
		--display-name "agnvenv (Python 3.9)"

lab: ## To start a Jupyter Lab server
ifndef SPS_HOME
		@source setup.sh
endif
	jupyter lab --notebook-dir=.

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	# @./bin/help.sh

.PHONY: run test kernel lab install docs docsimg qt help mypy alltest
