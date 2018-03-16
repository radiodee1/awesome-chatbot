#!/usr/bin/python3

import os
import sys

epochs=2 #* 24
files=17
big_file= False

#print(sys.argv)
print()
print('first arg = files')
print('next arg  = epochs')

if len(sys.argv) > 1:
    files = int(sys.argv[1]) - 1
    pass

if len(sys.argv) > 2:
    epochs = int(sys.argv[2]) - 1

print('\nargs (files, epochs):',files,epochs)

for i in range(epochs):
    for j in range(files):
        print('### epoch', i+1,'###')
        os.system('echo "### epoch num: '+ str(i+1) + ' , step num: '+ str(j+1) + ' ### " >> saved/progress.txt')
        if not big_file:
            os.system('./do_make_rename_train.sh ' + str(j + 1))
        else:
            os.system('./do_make_rename_train.sh ')

        os.system('./do_launch_categorical.sh ' + '--mode=train --printable=[epoch:' + str(i+1)+',file:'+ str(j+1)+ ']')
