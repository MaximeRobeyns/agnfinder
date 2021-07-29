#!/usr/bin/env bash
set -euo pipefail

# This is a helper script for writing documentation locally.
#
# If you are running this for the first time, please ensure that you have built
# the docker image.
#
# Check with
# >>> docker images | grep agnfinderdocs
# If not, run
# >>> make img
#
# With the sphinx image, run this script and open localhost:8081 to see the
# results. The sphinx docker image will watch files for changes under the
# ./source directory so when you make changes to the .rst files, the web
# preview will update.
#
# You can stop the running docker image with
# >>> docker stop agnfinderdocs
# or
# >>> docker stop $(docker ps -aq)

docker run --rm -v $(pwd)/source:/docs/source -v $(pwd)/build:/docs/build \
    --name agnfinderdocs \
    -p 8081:8080 agnfinderdocs watch
