#!/usr/bin/env python3

import argparse
import os
import sys

import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('--aiml', default='../hello.aiml', help='name of input aiml file.')

args = parser.parse_args()

folder = args.aiml.split('/')[:-1]
folder = '/'.join(folder)
print(folder)

tree = ET.parse(args.aiml)
root = tree.getroot()

tr_fr = open(folder + '/train.big.from', 'w')
tr_to = open(folder + '/train.big.to', 'w')

test_fr = open(folder + '/test.big.from', 'w')
test_to = open(folder + '/test.big.to', 'w')

val_fr = open(folder + '/valid.big.from', 'w')
val_to = open(folder + '/valid.big.to', 'w')

for x in root:

    for e in x:
        if e.tag == 'pattern':
            e.text = e.text.strip().lower()
            print('from:',e.text)
            tr_fr.write(e.text + '\n')
            test_fr.write(e.text + '\n')
            val_fr.write(e.text + '\n')
        if e.tag == 'template':
            e.text = e.text.strip().lower()
            print('to:',e.text)
            tr_to.write(e.text + '\n')
            test_to.write(e.text + '\n')
            val_to.write(e.text + '\n')

tr_fr.close()
tr_to.close()

test_fr.close()
test_to.close()

val_fr.close()
val_to.close()

os.chdir(folder)

os.system('zip chat_aiml train.big.from train.big.to test.big.from test.big.to valid.big.from valid.big.to ')

