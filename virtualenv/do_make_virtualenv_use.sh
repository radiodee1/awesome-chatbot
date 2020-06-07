#!/usr/bin/env bash

export VIRTUALENVWRAPPER_PYTHON=$(which python3.7)
source $(which virtualenvwrapper.sh)
workon chatbot

## type `deactivate` to exit ##