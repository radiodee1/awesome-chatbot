#!/usr/bin/env bash

DOCKER_BUILDKIT=1

./do_find_credentials.sh

cd ../


docker build --tag awesome_v7:1.0 -f ${PWD}/Dockerfile.armv7 .

cd docker

./do_launch_armv7.sh

#docker run -p 8001:8001 --entrypoint "./game_sr.py" awesome:1.0