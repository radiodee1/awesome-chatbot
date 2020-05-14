#!/usr/bin/env bash

DOCKER_BUILDKIT=1

cd ../..

echo ${PWD}

docker build --tag awesome:1.0 ./docker/docker_test_armv7/
#docker run --publish 8000:8080 awesome:1.0
./do_make_docker_arm32v7.sh

