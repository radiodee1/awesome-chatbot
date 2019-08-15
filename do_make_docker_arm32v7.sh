#!/usr/bin/env bash

docker pull emacski/tensorflow-serving:1.14.0-arm32v7

TESTDATA="$(pwd)/saved/t2t_train"
MODEL_NAME="chat_10"
EXPORT_NUM="1564940385"

#cp  -R --parent $TESTDATA/$MODEL_NAME/export/$EXPORT_NUM/* $TESTDATA/$MODEL_NAME/.

#ls -hal $TESTDATA/$MODEL_NAME

#docker run --mount type=bind,src=${TESTDATA}/${MODEL_NAME}/export/${EXPORT_NUM}/,dst=/${MODEL_NAME}  --entrypoint ls emacski/tensorflow-serving:1.14.0-arm32v7  -hal chat_10
docker run --mount type=bind,src=${TESTDATA}/${MODEL_NAME}/,dst=/${MODEL_NAME}  --entrypoint ls emacski/tensorflow-serving:1.14.0-arm32v7  -hal chat_10/export/




docker run -t --rm -p 8500:8500 \
    --mount type=bind,src=${TESTDATA}/${MODEL_NAME}/export/${EXPORT_NUM}/,dst=/${MODEL_NAME} \
    -e MODEL_NAME=$MODEL_NAME -e MODEL_BASE_PATH="" \
    --entrypoint tensorflow_model_server emacski/tensorflow-serving:1.14.0-arm32v7 \
    --port=8500 --model_name=${MODEL_NAME} --model_base_path=/$MODEL_NAME/export/${EXPORT_NUM}/ \
    #emacski/tensorflow-serving:1.14.0-arm32v7  # &

#    --mount type=bind,src=${TESTDATA}/${MODEL_NAME}/export/${EXPORT_NUM}/,dst=/${MODEL_NAME} \
exit

docker run --rm -ti -p 8500:8500 \
    -v ${TESTDATA}/${MODEL_NAME}/export/${EXPORT_NUM}:/models/${MODEL_NAME}/1 \
    -e MODEL_NAME=${MODEL_NAME} \
    emacski/tensorflow-serving:1.14.0-arm32v7
#cd $TESTDATA/$MODEL_NAME/.