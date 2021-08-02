#!/bin/bash

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

PYTHON=$1

# Create dependencies directory
if [[ ! -d ./deps ]]; then
    mkdir deps
fi

# verify that GNU unzip is installed to extract FSPS
command -v tar >/dev/null 2>&1 || {
    echo >&2 "tar is required to extract fsps but was not found. Aborting."
    exit 1
}

# This is required before installing fsps
export SPS_HOME=$(pwd)/deps/fsps

# Clone FSPS source code to dependencies
if [[ ! -d ./deps/fsps ]]; then
    git clone https://github.com/cconroy20/fsps.git $SPS_HOME

    # Alternative: might be better to download a stable release...
    # wget -O ./deps/fsps.tar.gz \
    #     https://github.com/cconroy20/fsps/archive/refs/tags/v3.2.tar.gz
    # tar xzf ./deps/fsps.tar.gz -C ./deps
    # mv ./deps/fsps-3.2 ./deps/fsps
    # rm ./deps/fsps.tar.gz
fi

# Create virtual environment if it doesn't exist
if [[ ! -d ./agnvenv ]]; then
    $PYTHON -m venv agnvenv
fi

# activate virtual environment
source agnvenv/bin/activate

# upgrade pip if available
python -m pip install --upgrade pip

# install dependencies
pip install -r requirements.txt

# copy custom sedpy filters to sedpy install location
SEDPY=$(echo -e "import sedpy\nprint(sedpy.__file__)\n" | python | xargs dirname)
cp -n ./filters/* $SEDPY/data/filters/

# install agnfinder itself
pip install -e .
