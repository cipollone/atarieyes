# Inside the server, I'll use docker, instead of a virtual environment.
# This image is just for development:
#   I'm not copying the source files nor installing the python dependencies
#   because I expect a bind mount on the home. Docker is only used for a
#   proper Tensorflow installation.

FROM tensorflow/tensorflow:2.1.0-gpu-py3

# Prepare user and home
RUN mkdir /home/ubuntu

RUN groupadd ubuntu && \
useradd -g ubuntu -d /home/ubuntu ubuntu && \
chown ubuntu:ubuntu /home/ubuntu
ENV HOME=/home/ubuntu
ENV SHELL=/bin/bash

WORKDIR /home/ubuntu/

# Apt
RUN apt-get -y update && \
apt-get -y upgrade && \
apt-get -y autoremove

# Basics
RUN apt-get -y install bash-completion git tmux \
	python3-pip python3-venv
# Opencv (project dependency)
RUN apt-get -y install libsm6 libxrender-dev
# Flloat (project dependency)
RUN apt-get -y install graphviz

# Python tools
RUN pip3 install --no-cache-dir -U wheel setuptools pip

# Go
USER ubuntu

ENTRYPOINT ["bash", "-l"]
