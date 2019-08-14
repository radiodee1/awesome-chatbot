#!/usr/bin/env bash

docker pull emacski/tensorflow-serving:1.14.0-arm32v7

TESTDATA="$(pwd)/saved/t2t_train/"
MODEL_NAME="chat_10"
EXPORT_NUM="1564940385"

cp  -R --parent $TESTDATA/$MODEL_NAME/export/$EXPORT_NUM/* $TESTDATA/.

ls -hal $TESTDATA/$MODEL_NAME

#docker run -v $TESTDATA --entrypoint ls emacski/tensorflow-serving:1.14.0-arm32v7 -hal

cd $TESTDATA/$MODEL_NAME/.

docker run -t --rm -p 8500:8500 \
    --mount "src=$TESTDATA/$MODEL_NAME/,dst=/$MODEL_NAME" \
    -e MODEL_NAME=$MODEL_NAME -e MODEL_BASE_PATH="" \
    --entrypoint tensorflow_model_server emacski/tensorflow-serving:1.14.0-arm32v7 \
    --port=8500 --model_name=${MODEL_NAME} --model_base_path=/$MODEL_NAME \
    #emacski/tensorflow-serving:1.14.0-arm32v7  # &

