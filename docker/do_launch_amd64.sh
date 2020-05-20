#!/usr/bin/env bash

DOCKER_BUILDKIT=1
ulimit -c unlimited

cd ../

if [[ -z "${ALSA_CARD}" ]]; then
    ALSA_CARD='PCH'
fi

echo ${ALSA_CARD}
echo ${#}
echo ${@}
if [[ "${#}" == "1" ]]; then
    ENTRY_POINT=${1}
elif [[ -z "${ENTRY_POINT}" ]]; then
    ENTRY_POINT=${@}
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
    --device /dev/snd --group-add audio --env ALSA_CARD=${ALSA_CARD} \
    --name awe_64 \
    --env CSE_ID=${CSE_ID} --env API_KEY=${API_KEY} \
    --env DEBIAN_FRONTEND=noninteractive \
    --env CREDENTIALS="${ENTRY_POINT}" -ti \
    --env GOOGLE_APPLICATION_CREDENTIALS=/app/${CREDENTIALS} \
    awesome:1.0 \
    --entrypoint ${ENTRY_POINT}