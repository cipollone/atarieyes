docker run --rm -it --hostname=eyes2dock --gpus all \
-p 30013:30013 -p 30014:30014 \
--mount type=bind,src=/home/ubuntu/,dst=/home/ubuntu/ eyes2dock
