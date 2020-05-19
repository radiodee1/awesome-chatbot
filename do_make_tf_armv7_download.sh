#!/usr/bin/env bash


wget --no-check-certificate https://github.com/radiodee1/pytorch-arm-builds/raw/master/torch-1.4.0a0%2B7f73f1d-cp37-cp37m-linux_armv7l.whl

wget --no-check-certificate https://www.piwheels.org/simple/opencv-python/opencv_python-4.1.1.26-cp37-cp37m-linux_armv7l.whl #sha256=f600ca8c1ba09c8f974b322bfd23662971cb051ab41ff2278042622c201c6a2a

wget --no-check-certificate https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.14.0-buster/tensorflow-1.14.0-cp37-none-linux_armv7l.whl


#wget --no-check-certificate  http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-armv7l.sh -O miniconda.sh