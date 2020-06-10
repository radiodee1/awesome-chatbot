#!/usr/bin/env python3

import argparse
import os
import sys
import math

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
parser.add_argument('--char-tab', default=False, action='store_true', help='tab divider character for saved files.')
parser.add_argument('--aiml', default=None, type=str, help='name of input aiml file.')
parser.add_argument('--tab', default=None, type=str, help='name of input tab file.')
parser.add_argument('--basename', default='../train.big', type=str, help='basename for target files.')
parser.add_argument('--tempname', default='../train.base', type=str, help='temporary base name.')
parser.add_argument('--write-over', default=False, action='store_true', help='write over files in place.')
parser.add_argument('--ratio', default=.5, type=float, help='ratio of inserted lines to original lines.')
parser.add_argument('--eol', default=False, action='store_true', help='strip or insert eol markers.')
parser.add_argument('--zip', default=False, action='store_true', help='zip files.')
parser.add_argument('--test', default=False, action='store_true', help='test some function.')

args = parser.parse_args()

folder_t = args.tempname.split('/')[:-1]
folder_temp = '/'.join(folder_t) + '/'

folder_t = ''
folder_b = ''

if args.test:
    mixlist = read_tab_multiline(args.tab, split_ret=True, strip_ret=True)
    print(mixlist)
    exit()

if args.aiml and not args.tab:
    mixlist = read_aiml(args.aiml)

if args.tab and not args.aiml:
    mixlist = read_tab(args.tab)

if args.ratio > 0.5:
    print('ratio value larger than 0.5 -- distortion')
    #exit()

char = '\n'
if args.char_tab:
    char = '\t'

tr_fr = open(folder_t + args.tempname + '.from', 'w')
tr_to = open(folder_t + args.tempname + '.to', 'w')
tr_qu = open(folder_t + args.tempname + '.ques', 'w')
##############

src_tr_fr = open(folder_b + args.basename + '.from', 'r')
src_tr_to = open(folder_b + args.basename + '.to', 'r')

f_fr = src_tr_fr.readlines()
f_to = src_tr_to.readlines()

if len(f_fr) != len(f_to):
    print("input files don't match")
    exit()
else:
    pass

############
w = len(mixlist)
z = len(f_fr)
r = 1 - args.ratio
x = ((z ) / r) - z ## num lines added
c = x + z          ## tot num lines
d = c / x          ## interval?
print(r, x, z, c, d)
if args.ratio > 0.5:
    c = math.ceil(c / args.ratio)
    d = math.ceil(d / args.ratio)
    print(c,d)
############
num = 0
mix = 0
for i in range(int(c)):
    if i % int(d) == 0:
        temp = mix % w
        mixlist[temp] = set_eol(mixlist[temp], args.eol)
        tr_fr.write(mixlist[temp][0] + char)
        tr_to.write(mixlist[temp][1] + char)
        tr_qu.write(char)
        mix += 1
    else:
        a = f_fr[num % z].strip()
        b = f_to[num % z].strip()
        a, b = set_eol([a,b], args.eol)
        tr_fr.write(a + char)
        tr_to.write(b + char)
        tr_qu.write(char)
        num += 1


############
tr_fr.close()
tr_to.close()

################

src_tr_fr.close()
src_tr_to.close()

os.chdir(folder_temp)

if args.write_over:
    tname = args.tempname.split('/')[-1]
    bname = args.basename.split('/')[-1]
    os.system('mv ' + tname + '.from ' + bname + '.from' )
    os.system('mv ' + tname + '.to ' + bname + '.to' )
    os.system('mv ' + tname + '.ques ' + bname + '.ques' )

    if args.zip:
        bname_zip = bname.split('.')[-1]
        os.system('zip chat_' + bname_zip + ' ' + bname + '.from ' + bname + '.to ' + bname + '.ques ')
else:
    tname = args.tempname.split('/')[-1]
    tname_zip = tname.split('.')[-1]
    if args.zip:
        os.system('zip chat_' + tname_zip + ' ' + tname + '.from ' + tname + '.to ' + tname + '.ques ')