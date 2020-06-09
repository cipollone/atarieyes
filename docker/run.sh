docker run --rm -it --hostname=eyes2dock --gpus all --network host \
--mount type=bind,src=/home/ubuntu/,dst=/home/ubuntu/ eyes2dock
