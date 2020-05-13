#!/usr/bin/env bash

DOCKER_BUILDKIT=1

cd ../..

echo ${PWD}

docker build --tag awesome:1.0 ./docker/docker_test_x86_64/
#docker run --publish 8000:8080 awesome:1.0
