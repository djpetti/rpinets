#!/bin/bash

# Check the number of arguments.
if [ "$#" -ne 1 ] || ! [ -d "$1" ]; then
  echo "Usage: $0 TRAINING_DATA_DIR" >&2
  exit 1
fi

sudo nvidia-docker run --rm -ti -v "$(dirname "`pwd`/../../..")":/job_files \
    -v "$1":/home/theano/training_data --net=host \
    djpetti/rpinets-theano:imagenet_server /bin/bash
