#!/usr/bin/python3

import os

epochs=2
files=17

for i in range(epochs):
    for j in range(files):
        print('### epoch', i+1,'###')
        os.system('./do_make_rename_train.sh ' + str(j + 1))
        os.system('./do_launch_model.sh')