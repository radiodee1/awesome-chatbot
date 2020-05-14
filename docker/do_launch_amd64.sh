#!/usr/bin/env bash

DOCKER_BUILDKIT=1

cd ../

echo ${PWD}

#docker build --tag awesome:1.0 -f ${PWD}/Dockerfile.amd64 .

docker run -p 8001:8001 --entrypoint "./game_sr.py" awesome:1.0