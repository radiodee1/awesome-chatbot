#!/usr/bin/env bash

docker pull emacski/tensorflow-serving:1.14.0-arm32v7

TESTDATA="$(pwd)/saved/t2t_train/chat_10/export/1564940385/"

docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA" \
    -e MODEL_NAME=chat \
    emacski/tensorflow-serving:1.14.0-arm32v7   &