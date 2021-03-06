#!/bin/bash

sudo docker run -ti --device /dev/nvidia0:/dev/nvidia0 --device \
  /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  --net=host djpetti/tensorflow /run_tests.sh
