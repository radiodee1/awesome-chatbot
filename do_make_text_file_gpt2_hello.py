#!/usr/bin/python3

import argparse
import os
import sys
import math

import xml.etree.ElementTree as ET

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
    'Christopher'
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

ques1 = '''
Q: My name is {}.
A: Hello {}.

Q: What is my name?
'''
ans1 = '''
A: {}.
'''


mixlist = []

def read_aiml(name):
    tree = ET.parse(name)
    root = tree.getroot()
    somelist = []
    for x in root:
        tmp = []
        for e in x:
            if e.tag == 'pattern':
                e.text = e.text.strip().lower()
                #print('from:', e.text)
                tmp.append(e.text)

            if e.tag == 'template':
                e.text = e.text.strip().lower()
                #print('to:', e.text)
                tmp.append(e.text)

        somelist.append(tmp)
    return somelist

def read_tab(name):
    with open(name,'r') as f:
        somelist = []
        ff = f.readlines()
        for i in ff:
            #tmp = []
            i = i.strip()
            i = i.split('\t')
            #print(i[0],'<',i[1])
            somelist.append([i[0].strip(), i[1].strip()])
    return somelist

def read_tab_multiline(name, split_ret=False, strip_ret=False):
    with open(name,'r') as f:
        somelist = []
        returnlist = []
        tmp = ''
        ff = f.readlines()
        for i in ff:
            #tmp = []
            if '\t' in i :
                i = i.split('\t')
                for j in range(len(i)):
                    tmp = tmp + i[j]
                    if strip_ret: tmp = tmp.strip()
                    somelist.append(tmp)
                    tmp = ''
                if split_ret:
                    if strip_ret: tmp = tmp.strip()
                    somelist.append(tmp)
                    tmp = ''
            else:
                tmp = tmp + i
        num = 0
        a_part = None
        b_part = None
        for i in somelist:
            if num % 2 == 0:
                a_part = i
            else:
                b_part = i
                returnlist.append([a_part, b_part])
            num += 1
    return returnlist

def set_eol(i, do_eol):
    a = i[0]
    b = i[1]
    if do_eol:
        if a.endswith('eol') or b.endswith('eol'):
            pass
        else:
            a += ' eol'
            b += ' eol'
    else:
        a = a.replace('eol', '')
        b = b.replace('eol', '')
    return a, b

parser = argparse.ArgumentParser()
parser.add_argument('--basename', default='../train.base', type=str, help='basename for target files.')
parser.add_argument('--mult', default=10, type=int, help='number of repetitions.')
parser.add_argument('--single', default=False, action='store_true', help='store single file.')
args = parser.parse_args()

print(len(namelist), 'num of names.')

if args.single:
    with open(args.basename + '.txt', 'w') as f:
        for num in range(args.mult):
            for i in namelist:
                j = ques1 + ans1
                j = j.format(i,i,i)
                if num == 0 and False: print(j)
                f.write(j+'\t')

else:
    with open(args.basename + '.from', 'w') as f_from, open(args.basename + '.to', 'w') as f_to:
        for num in range(args.mult):
            for i in namelist:
                f_from.write(ques1.format(i,i) + '\t')
                f_to.write(ans1.format(i) + '\t')
