#!/usr/bin/env bash

DOCKER_BUILDKIT=1

cd ../

echo ${#}
if [[ "${#}" == "1" ]]; then
    ENTRY_POINT=${1}
fi
echo ${PWD}
export CREDENTIALS=`cat credentials.txt`
echo ${CREDENTIALS}

if [[ -z "${ENTRY_POINT}" ]]; then
    ENTRY_POINT="/bin/bash"
    #ENTRY_POINT="jackd -R -d alsa -d hw:1"
fi

#export ENTRY_POINT=./bot/game_sr.py
#export ENTRY_POINT=/bin/bash

echo ${ENTRY_POINT}

export CSE_ID=$(cat cse_id.txt)
export API_KEY=$(cat api_key.txt)

docker run -p 8001:8001 --mount type=bind,src=${PWD}/,dst=/app/. \
    --device /dev/snd --group-add audio --env ALSA_CARD="PCH" \
    --env CSE_ID=${CSE_ID} --env API_KEY=${API_KEY} \
    --env DEBIAN_FRONTEND=noninteractive \
    --env CREDENTIALS="${ENTRY_POINT}" -ti \
    --env GOOGLE_APPLICATION_CREDENTIALS=/app/${CREDENTIALS} \
    --entrypoint "${ENTRY_POINT}" awesome:1.0