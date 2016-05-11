#!/bin/bash

sudo docker run -ti --device /dev/nvidia0:/dev/nvidia0 --device \
  /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  -v "$(dirname "`pwd`")":/home/theano/research --net=host djpetti/theano /bin/bash
