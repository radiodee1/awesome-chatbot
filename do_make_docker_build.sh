#!/usr/bin/env bash

echo $1

ROOT=$PWD
echo $ROOT

cd $1

docker build -t testapp:latest -f $ROOT/docker_t2t/Dockerfile --build-arg EXPORT=$1 .

