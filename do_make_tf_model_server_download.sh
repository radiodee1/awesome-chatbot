#!/usr/bin/env bash

cd raw

wget https://github.com/emacski/tensorflow-serving-arm/releases/download/v1.12.0/tensorflow_model_server-1.12.0-linux_armhf.tar.gz

echo unpack tarball and move executable to sensible location
echo /usr/local/bin

tar -xvf tensorflow_model_server-1.12.0-linux_armhf.tar.gz

