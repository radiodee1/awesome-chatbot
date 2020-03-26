#!/usr/bin/python3

import argparse
import os
import sys

import xml.etree.ElementTree as ET

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

parser = argparse.ArgumentParser()
parser.add_argument('--aiml', default=None, type=str, help='name of input aiml file.')
parser.add_argument('--tab', default=None, type=str, help='name of input tab file.')
parser.add_argument('--basename', default='../train.big', type=str, help='basename for target files.')
parser.add_argument('--tempname', default='../train.base', type=str, help='temporary base name.')
parser.add_argument('--write_over', default=False, type=bool, help='write over files in place.')
parser.add_argument('--ratio', default=.5, type=float, help='ratio of inserted lines to original lines.')

args = parser.parse_args()

folder_t = args.tempname.split('/')[:-1]
folder_t = '/'.join(folder_t) + '/'
print(folder_t)

folder_b = args.basename.split('/')[:-1]
folder_b = '/'.join(folder_b) + '/'
print(folder_b)

folder_t = ''
folder_b = ''

if args.aiml and not args.tab:
    mixlist = read_aiml(args.aiml)

if args.tab and not args.aiml:
    mixlist = read_tab(args.tab)

if args.ratio > 0.5:
    print('no ratio value larger than 0.5')
    exit()


tr_fr = open(folder_t + args.tempname + '.from', 'w')
tr_to = open(folder_t + args.tempname + '.to', 'w')

#test_fr = open(folder_t + args.tempname + '.from', 'w')
#test_to = open(folder_t + args.tempname + '.to', 'w')

#val_fr = open(folder_t + args.tempname + '.from', 'w')
#val_to = open(folder_t + args.tempname + '.to', 'w')

##############

src_tr_fr = open(folder_b + args.basename + '.from', 'r')
src_tr_to = open(folder_b + args.basename + '.to', 'r')

#src_test_fr = open(folder_b + args.basename + '.from', 'r')
#src_test_to = open(folder_b + args.basename + '.to', 'r')

#src_val_fr = open(folder_b + args.basename + '.from', 'r')
#src_val_to = open(folder_b + args.basename + '.to', 'r')

############
print(mixlist)
z = len(mixlist)
r = 1 - args.ratio
x = ((z ) / r) - z
print(x)
c = x + z

print(r, z, c)
############

tr_fr.close()
tr_to.close()

#test_fr.close()
#test_to.close()

#val_fr.close()
#val_to.close()

################

src_tr_fr.close()
src_tr_to.close()

#src_test_fr.close()
#src_test_to.close()

#src_val_fr.close()
#src_val_to.close()
