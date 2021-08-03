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

# Outputs some help information

cat << EOF

AGNfinder: Detect AGN from photometry in XXL data.

Please use 'make target' where 'target' is one of:

    install     To install everything (warning, downloads ~1.5Gb of data)
    test        To run the program's tests (and verify the installation)
    run         To run the main sampling program
    docs        To compile the documentation (requires Docker)
    qt          To re-create the quasar templates (quasar and torus models)
    kernel      To setup a Jupyter kernel
    lab         To start a Jupyter Lab server

EOF
