#!/bin/bash

PYTHON=$1

# Create dependencies directory
if [[ ! -d ./deps ]]; then
    mkdir deps
fi

# verify that wget is installed
command -v wget >/dev/null 2>&1 || {
    echo >&2 "wget is required to download fsps but was not found. Aborting."
    exit 1
}

# verify that GNU unzip is installed to extract FSPS
command -v tar >/dev/null 2>&1 || {
    echo >&2 "tar is required to extract fsps but was not found. Aborting."
    exit 1
}

# Clone FSPS source code to dependencies
if [[ ! -d ./deps/fsps ]]; then
    wget -O ./deps/fsps.tar.gz \
        https://github.com/cconroy20/fsps/archive/refs/tags/v3.2.tar.gz
    tar xzf ./deps/fsps.tar.gz -C ./deps
    mv ./deps/fsps-3.2 ./deps/fsps
    rm ./deps/fsps.tar.gz
fi

# This is required before installing fsps
export SPS_HOME=$(pwd)/deps/fsps

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
SEDPY=$(echo -e "import sedpy\nprint(sedpy.__file__)\n" | $PYTHON | xargs dirname)
cp -n ./filters/* $SEDPY

# install agnfinder itself
pip install -e .
