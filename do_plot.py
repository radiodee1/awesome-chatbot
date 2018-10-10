#!/usr/bin/python3.6

from __future__ import unicode_literals, print_function, division

import sys
sys.path.append('..')
from io import open
import unicodedata
import string
import re
import random
import os
import time
import datetime
import math
import matplotlib.pyplot as plt
import argparse
import json
import cpuinfo
from model.settings import hparams
#from model import  tokenize_weak
import glob

if __name__ == '__main__':
    #os.chdir('/' + '/'.join(sys.argv[0].split('/')[:-1]))
    parser = argparse.ArgumentParser(description='Plot some NMT values.')
    parser.add_argument('--files', help='File glob for plotting. Must be json files!!')
    parser.add_argument('--title', help='Graph title.')

    args = parser.parse_args()
    args = vars(args)
    print(args)

    do_filelist = False
    do_title_graph = False

    if args['files'] is not None:
        do_filelist = True

    if args['title'] is not None:
        do_title_graph = True

    arg_filename = '/'.join( hparams['save_dir'].split('/')[1:]) + '/' + 'test*.json'
    arg_title = 'Autoencode vs. Steps'

    if do_filelist:
        arg_filename = str(args['files'])

    if do_title_graph:
        arg_title = str(args['title'])

    arg_filelist = arg_filename.split(',')
    arg_glob_list = []
    for i in arg_filelist:
        print(i,'use for plot')
        arg_glob_list.extend(glob.glob(i))

    print(arg_glob_list)
    arg_list = []
    for i in arg_glob_list:
        if os.path.isfile(i):
            with open(i, 'r') as z:
                sublist = []
                j = json.loads(z.read())
                for k in j:
                    sublist.append((int(k), float(j[k])))
                sublist.sort(key=lambda tuple: tuple[0])
                print(sublist)
                arg_list.append(sublist)

    print(arg_list)
