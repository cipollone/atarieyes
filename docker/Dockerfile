# Inside the server, I'll use docker, instead of pipenv.

# This image is just for development:
# I'm not copying the source files, because I'm expecting
# a bind mount somewere inside the container.

FROM tensorflow/tensorflow:2.1.0-gpu-py3

# Prepare user and home
RUN mkdir /home/ubuntu

RUN groupadd ubuntu && \
useradd -g ubuntu -d /home/ubuntu ubuntu && \
chown ubuntu:ubuntu /home/ubuntu
ENV HOME=/home/ubuntu

WORKDIR /home/ubuntu/

# Sw
RUN apt-get -y update && \
apt-get -y upgrade

# Project dependencies

# Basics
RUN apt-get -y install python3-pip bash-completion tmux git 
# Opencv
RUN apt-get -y install libsm6 libxrender-dev

# Basics
RUN pip3 --no-cache-dir install -U wheel setuptools pip
# [dev-packages]
RUN pip3 --no-cache-dir install -U ipython flake8 pipenv-setup
# [packages] (tensorflow already there)
RUN opencv-python numpy gym[atari] keras-rl

# Go
USER ubuntu

ENV PATH="/home/ubuntu/bin:/home/ubuntu/.local/bin:$PATH"

ENTRYPOINT ["bash"]
