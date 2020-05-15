#!/usr/bin/env bash

cd ../

echo ${PWD}
echo ${HOME}
cp ${HOME}/bin/awesome-sr-*.json ${PWD}/.

ls awesome-sr-*.json > credentials.txt

cat credentials.txt

cd docker