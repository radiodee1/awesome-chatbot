#!/usr/bin/env bash

sudo apt-get update \
    && sudo apt-get install -y mpg123 python3-pyaudio vim python3 python3-pip curl mpg321 \
    alsa-utils alsa-base libasound2-plugins portaudio19-dev

echo 'deb [arch=amd64] https://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal' |  sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
sudo curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update \
    && sudo apt-get install -y tensorflow-model-server