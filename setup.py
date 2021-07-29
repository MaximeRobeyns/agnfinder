# AGNfinder: Detect AGN from photometry in XXL data.
#
# Copyright (C) 2021 Maxime Robeyns <maximerobeyns@gmail.com>
# Copyright (C) 2019-20 Mike Walmsley <walmsleymk1@gmail.com>
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

import pathlib
from setuptools import setup
from setuptools import find_packages

wd = pathlib.Path(__file__).parent.resolve()

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()
    install_requires = list(filter(lambda s: '=' in s, install_requires))

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='agnfinder',
    version='0.0.2',
    author='Maxime Robeyns',
    author_email='maximerobeyns@gmail.com',
    description='Find AGN in XMM/Euclide photometry',
    long_description=long_description,
    url='https://github.com/MaximeRobeyns/agnfinder',
    license='GPLv3',
    install_requires=install_requires,
    packages=find_packages(exclude=['tests', 'old_agnfinder', 'old_tests']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU GPLv3 License',
        'Operating System :: OS Independent',
    ],
    project_urls={
        'Documentation': 'https://github.com/MaximeRobeyns/agnfinder',
        'Bug Reports': 'https://github.com/MaximeRobeyns/agnfinder/issues',
        'Source': 'https://github.com/MaximeRobeyns/agnfinder',
    },
)
