#!/bin/bash

# WARNING: this script is deprecated (from pre-fork agnfinder) and is left in
# only for reference.

CUBES=20  # zeus cores
SAMPLES_PER_CUBE=10000  # 14 mins for 100k, 140 mins for 1m, etc.
ZMAX=1.  # okay to change this, but never change the 4.0 in simulate_samples.py

ZSTEP=$(echo "scale=4;$ZMAX/$CUBES" | bc)

LASTZ=0
for Z in $(seq $ZSTEP $ZSTEP $ZMAX)
do
    echo "Starting cube for redshift slice: ${LASTZ} to ${Z}"
    nohup python agnfinder/simulation_samples.py ${SAMPLES_PER_CUBE} --z-min ${LASTZ} --z-max ${Z} --save-dir data/cubes/new > "data/cubes/new/${SAMPLES_PER_CUBE}_${LASTZ}_${Z}.log" &
    LASTZ=$Z
    echo "Cube complete"
done
