#!/usr/bin/env bash

DOCKER_BUILDKIT=1

./do_find_credentials.sh

cd ../

docker build --tag awesome:1.0 -f ${PWD}/Dockerfile.amd64 .

cd docker

./do_launch_amd64.sh

