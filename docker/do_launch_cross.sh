#!/bin/bash

DOCKER_BUILDKIT=1
DOCKER_LOCAL_ARGS=""
cd ../

echo ${PWD}

if [[ -z "${ALSA_CARD}" ]]; then
    ALSA_CARD='PCH'
fi

###########################


########################################

Container_ID2=$(docker ps -aqf 'name=^awe_cross_v7$')


if [[ -z "${Container_ID2}" ]]; then
    Container_ID2="xxx"
fi

result=$( docker inspect -f {{.State.Running}} $Container_ID2)

echo $Container_ID2
echo $result

if [[  -z "$result" ]]; then
    echo fail
    result="false"
else
    echo pass
    result="true"
fi

echo "result is" $result

#result="true"

if [[ $result = "true" ]]; then
    echo "awe_x7 is already running ${Container_ID2}"
    #docker restart ${Container_ID2}
    export DOCKER_LOCAL_ARGS="--link awe_cross_v7 ${DOCKER_LOCAL_ARGS}"
    #exit
else
    export DOCKER_LOCAL_ARGS="--name awe_cross_v7 ${DOCKER_LOCAL_ARGS}"
    echo "new arm instance must be started."
fi
###############################################

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

echo ${ENTRY_POINT}

export CSE_ID=$(cat cse_id.txt)
export API_KEY=$(cat api_key.txt)

echo ${DOCKER_DAEMON_ARGS}
echo "${DOCKER_LOCAL_ARGS} <<--"

docker run -p 8002:8002  --mount type=bind,src=${PWD}/,dst=/app/. \
    --device /dev/snd:/dev/snd --group-add audio --env ALSA_CARD=${ALSA_CARD} \
    --privileged ${DOCKER_LOCAL_ARGS} \
    --env CSE_ID=${CSE_ID} --env API_KEY=${API_KEY} \
    --env DEBIAN_FRONTEND=noninteractive \
    --env CREDENTIALS="${ENTRY_POINT}" -ti \
    --env GOOGLE_APPLICATION_CREDENTIALS=/app/${CREDENTIALS} \
    --env DOCKER_DAEMON_ARGS="" \
    awesome_x7/dind:1.0 \
    --entrypoint ${ENTRY_POINT}

# --env DOCKER_HOST="tcp://0.0.0.0:2375" \
