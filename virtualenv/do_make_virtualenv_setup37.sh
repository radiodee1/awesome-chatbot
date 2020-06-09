#!/usr/bin/env bash

# sudo may not be needed here
sudo pip3 install --user virtualenv
sudo pip3 install --user virtualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
mkdir -p $WORKON_HOME
export VIRTUALENVWRAPPER_PYTHON=$(which python3.7)
source $(which virtualenvwrapper.sh)

mkvirtualenv chatbot37 --python $(which python3.7)
