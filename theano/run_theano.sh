#!/bin/bash

# Check the number of arguments.
if [ "$#" -ne 1 ] || ! [ -d "$1" ]; then
  echo "Usage: $0 TRAINING_DATA_DIR" >&2
  exit 1
fi

sudo nvidia-docker run --rm -ti -v "$(dirname "`pwd`")":/home/theano/rpinets \
    -v "$1":/home/theano/training_data --net=host cde85a34ff83 /bin/bash
