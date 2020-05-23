#!/usr/bin/env bash

DOCKER_BUILDKIT=1

#export DOCKER_CLI_EXPERIMENTAL=enabled
./do_find_credentials.sh

cd ..

#iptables -L
#docker run --rm --privileged multiarch/qemu-user-static:arm32v7 --reset -p yes


docker  build  --tag awesome_x7/dind:1.0 -f ${PWD}/Dockerfile.cross .

cd docker

exit
./do_launch_armv7.sh

## -v /var/run/docker.sock:/var/run/docker.sock