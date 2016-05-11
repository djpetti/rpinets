FROM djpetti/theano:latest
MAINTAINER Daniel Petti

# Install packages that we need to run code.
RUN apt-get update && apt-get install -y python-opencv

# Make the display work right.
RUN echo "export DISPLAY=:0" >> /home/theano/.bashrc
