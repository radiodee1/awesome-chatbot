#!/usr/bin/env bash

echo $1

cd docker_t2t

docker build -t testapp:latest . --build-arg EXPORT=$1

