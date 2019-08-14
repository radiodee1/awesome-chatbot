#!/usr/bin/env bash

docker pull emacski/tensorflow-serving:1.14.0-arm32v7

TESTDATA="$(pwd)/saved/t2t_train"
MODEL_NAME="chat_10"
EXPORT_NUM="1564940385"

cp  -R --parent $TESTDATA/$MODEL_NAME/export/$EXPORT_NUM/* $TESTDATA/$MODEL_NAME/.

ls -hal $TESTDATA/$MODEL_NAME

docker run --mount type=bind,src=${TESTDATA}/,dst=/${MODEL_NAME}  --entrypoint ls emacski/tensorflow-serving:1.14.0-arm32v7  -hal *
exit

cd $TESTDATA/$MODEL_NAME/.

docker run -t --rm -p 8500:8500 \
    --mount type=bind,src=${TESTDATA}/,dst=/${MODEL_NAME} \
    -e MODEL_NAME=$MODEL_NAME -e MODEL_BASE_PATH="" \
    --entrypoint tensorflow_model_server emacski/tensorflow-serving:1.14.0-arm32v7 \
    --port=8500 --model_name=${MODEL_NAME} --model_base_path=/$MODEL_NAME \
    #emacski/tensorflow-serving:1.14.0-arm32v7  # &

