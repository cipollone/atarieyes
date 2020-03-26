# Inside the server, the user is `ubuntu`.
# ubuntu/docker is used to hold images and containers.

sudo service docker stop
sudo mount --bind /home/ubuntu/docker /var/lib/docker
sudo service docker start
