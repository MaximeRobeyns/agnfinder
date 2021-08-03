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

FROM python:alpine
WORKDIR /docs
RUN apk update && apk add --no-cache bash make libffi-dev openssl-dev inotify-tools
RUN pip install sphinx sphinx_rtd_theme live-server
RUN mkdir -p ./build/html ./build/doctrees ./source
COPY Makefile.docker Makefile
COPY make.bat make.bat
COPY docs.sh /usr/local/bin/docs.sh
ENV PORT=8080
EXPOSE 8080
ENTRYPOINT ["sh", "/usr/local/bin/docs.sh"]
