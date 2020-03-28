#!/usr/bin/python3

import argparse
import os
import sys
import math
import random

#import xml.etree.ElementTree as ET

namelist = [
    'David',
    'Elisabeth',
    'Daniel',
    'Sarah',
    'John',
    'Jane',
    'Mary',
    'Philip',
    'xxxx',
    'yyyy',
    'zzzz',
    'Tom',
    'Joseph',
    'Edward',
    'Christina',
    'Diana',
    'Joan',
    'Christopher',
    'James',
    'Juley',
    'Vincent',
    'Sara',
    'Dave',
    'Laura',
    'Donald',
    'Peter',
    'Charles',
    'Lawrence',
    'Bess',
    'Fido',
    'Spot'
]

ques1 = ['','', '']

ques1[0] = '''
Q: What is your name?
A: My name is {}.
Q: My name is {}?
A: Hello {}.

Q: What is my name?
'''
ans1 = '''
A: {}.
'''

ques1[1] = '''
Q: What is your name?
A: I'm {}.
Q: I'm {}?
A: Hello {}.

Q: What is my name?
'''

ques1[2] = '''
Q: I'm {}?
A: Hello {}.
Q: What is your name?
A: I'm {}.

Q: What is my name?
'''

mixlist = []



parser = argparse.ArgumentParser()
parser.add_argument('--basename', default='../train.base', type=str, help='basename for target files.')
parser.add_argument('--mult', default=10, type=int, help='number of repetitions.')
parser.add_argument('--single', default=False, action='store_true', help='store single file.')
parser.add_argument('--zip', default=False, action='store_true', help='zip files.')

args = parser.parse_args()

print(len(namelist), 'num of names.')

if args.single:
    with open(args.basename + '.txt', 'w') as f:
        z = 0
        for num in range(args.mult):
            for i in namelist:
                if z % 3 == 0 or z % 3 == 1:
                    k1 = i
                    while k1 == i:
                        k1 = namelist[random.randint(0, len(namelist) - 1)]
                    k2 = i
                    k3 = i
                    pass
                if z % 3 == 2:
                    k3 = i
                    while k3 == i:
                        k3 = namelist[random.randint(0, len(namelist) - 1)]
                    k2 = i
                    k1 = i
                    pass

                j = ques1[z % 3] + ans1
                j = j.format(k1, k2,k3,i)
                if num == 0 and False: print(j)
                f.write(j+'\t')
                z += 1

else:
    with open(args.basename + '.from', 'w') as f_from, open(args.basename + '.to', 'w') as f_to:
        z = 0
        for num in range(args.mult):
            for i in namelist:
                if z % 3 == 0 or z % 3 == 1:
                    k1 = i
                    while k1 == i:
                        k1 = namelist[random.randint(0, len(namelist) - 1)]
                    k2 = i
                    k3 = i

                    pass
                if z % 3 == 2:
                    k3 = i
                    while k3 == i:
                        k3 = namelist[random.randint(0, len(namelist) - 1)]
                    k2 = i
                    k1 = i
                    pass

                f_from.write(ques1[z % 3].format(k1, k2,k3) + '\t')
                f_to.write(ans1.format(i) + '\t')
                z += 1

if args.zip:
    folder_t = args.basename.split('/')[:-1]
    folder = '/'.join(folder_t) + '/'

    os.chdir(folder)

    tname = args.basename.split('/')[-1]
    tname_zip = tname.split('.')[-1]
    if args.zip:
        os.system('zip chat_hello_' + tname_zip + ' ' + tname + '.from ' + tname + '.to ' + tname + '.ques ')