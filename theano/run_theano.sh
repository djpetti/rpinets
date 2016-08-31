#!/bin/bash

sudo nvidia-docker run -ti -v "$(dirname "`pwd`")":/home/theano/research --net=host djpetti/rpinets-theano /bin/bash
