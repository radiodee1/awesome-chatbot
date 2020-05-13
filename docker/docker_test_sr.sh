#!/usr/bin/env bash

docker pull ubuntu:19.10

#docker LABEL ubuntu:19.10=mydoc

docker run --name mydoc ubuntu  echo "deb http://archive.canonical.com/ubuntu focal main" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get -y install mpg123 mpg321 python3-pyaudio  vim
#docker run ubuntu apt-get update
#docker run ubuntu apt-get update && apt-get -y install mpg123 mpg321 python3-pyaudio python-pyaudio vim
#docker exec mydoc -it -p 8080:80  ubuntu /bin/bash
