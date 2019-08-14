#!/usr/bin/env bash

docker pull emacski/tensorflow-serving:1.14.0-arm32v7

TESTDATA="$(pwd)/saved/t2t_train/"
MODEL_NAME="chat_10"
EXPORT_NUM="1564940385"

cp  -R $TESTDATA/$MODEL_NAME/export/$EXPORT_NUM/* $TESTDATA/$MODEL_NAME/$MODEL_NAME/.

#ls $TESTDATA/$MODEL_NAME

docker run -t --rm -p 8500:8500 \
    -v $TESTDATA \
    -e MODEL_NAME=$MODEL_NAME -e MODEL_BASE_PATH="" \
    emacski/tensorflow-serving:1.14.0-arm32v7  # &