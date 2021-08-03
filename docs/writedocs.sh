#!/usr/bin/env bash
set -euo pipefail

# This is a helper script for writing documentation locally.
# This is intended to be run from the parent directory (e.g. either through the
# provided Makefile `make docs` or as ./docs/writedocs.sh)
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

if [[ "$OSTYPE" == "darwin"* ]]; then
    OS=mac
else
    OS=other
fi

# does what it says on the tin
openUrl() {
    if [[ $OS == mac ]]; then
        open $1
    else
        # [[ -x $BROWSER ]] && exec "$BROWSER" "$1"
        path=$(which xdg-open || which gnome-open) && exec "$path" "$1" > /dev/null 2>&1 &
    fi
}

# First check whether the docker image exists
if [[ -z $(docker images | grep agnfinderdocs) ]]; then
    docker build -f Dockerfile -t agnfinderdocs .
fi

openUrl http://localhost:8081

docker run --rm -v $(pwd)/docs/source:/docs/source -v $(pwd)/docs/build:/docs/build \
    -v $(pwd)/Makefile.writing:/docs/Makefile \
    --name agnfinderdocs \
    -p 8081:8080 agnfinderdocs watch
