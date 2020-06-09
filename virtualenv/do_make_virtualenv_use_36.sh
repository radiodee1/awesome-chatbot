#!/usr/bin/env bash

export VIRTUALENVWRAPPER_PYTHON=$(which python3.6)
source $(which virtualenvwrapper.sh)
workon chatbot36

## type `deactivate` to exit ##