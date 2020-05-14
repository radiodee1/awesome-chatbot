#!/usr/bin/env bash

DOCKER_BUILDKIT=1

cd ../

echo ${PWD}
echo ${HOME}
cp ${HOME}/bin/awesome-sr-*.json ${PWD}/.

ls ${PWD}/awesome-sr-*.json > credentials.txt

cat credentials.txt

docker build --tag awesome:1.0 -f ${PWD}/Dockerfile.amd64 .

cd docker

./do_launch_amd64.sh

#docker run -p 8001:8001 --entrypoint "./game_sr.py" awesome:1.0