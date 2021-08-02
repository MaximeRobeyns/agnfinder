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
run:
ifndef SPS_HOME
		@source setup.sh
endif
	@python agnfinder/simulation/simulation.py

test:
ifndef SPS_HOME
		@source setup.sh
endif
	@python -m pytest tests

docs:
	@./docs/writedocs.sh

# Install agnfinder project locally
install:
	@./bin/install_with_venv.sh $(PYTHON)

# ipython kernel setup to run notebooks in venv
kernel:
	python -m ipykernel install --user --name agnvenv \
		--display-name "agnvenv (Python 3.9)"

.PHONY: kernel install docs
