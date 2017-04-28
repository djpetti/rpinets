FROM gcr.io/tensorflow/tensorflow:latest-devel-gpu
MAINTAINER Daniel Petti

# Install packages that we need to run code.
RUN apt-get update && apt-get install -y python-opencv graphviz

RUN pip install pyaml pydot-ng h5py
