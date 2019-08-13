#!/usr/bin/env bash

docker pull emacski/tensorflow-serving:1.14.0-arm32v7



TESTDATA="$(pwd)/saved/t2t_train/" #chat_10/export/1564940385/"

cp -r $TESTDATA/chat_10/export/1565716437/* $TESTDATA/chat_10/

docker run -t --rm -p 8500:8500 \
    -v "$TESTDATA" \
    -e MODEL_NAME=chat_10 -e MODEL_BASE_PATH="" \
    emacski/tensorflow-serving:1.14.0-arm32v7  # &