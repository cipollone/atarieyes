docker run --rm -it --hostname=tf2dock --gpus all -p 30013:30013 \
--mount type=bind,src=/home/ubuntu/,dst=/home/ubuntu/ tf2dock
