#!/usr/bin/python3

import os
import argparse
import sys

parser = argparse.ArgumentParser(
    description='zip tensor-2-tensor on your chat dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--basename', help='start zip operation on this and other files.')
parser.add_argument('--latest', action='store_true', help='choose latest checkpoint.')
parser.add_argument('--name', default='chat', help='run filename.')
parser.add_argument('--no-vocab', action='store_true', help='skip storing vocab files.')
args = parser.parse_args()

if args.basename == None and not args.latest:
    exit('what files to save??!!')

if not args.latest:
    path = args.basename.strip().split('/')[:-1]
    path = '/'.join(path)
else:
    path = 'saved/t2t_train/' + args.name + '/'

if args.basename is not None:
    basename = args.basename.strip().split('/')[-1]
    basename = basename.strip().split('.')[0:-1]
    basename = '.'.join(basename)
    print(basename)
    if not basename.startswith('model'):
        exit('bad model name')

vocabname = 'data/t2t_data/' + args.name + '/' + 'vocab.' + '*'

if not args.no_vocab:
    os.system('zip t2t_' + args.name + ' ' + vocabname)

print(path)

if os.path.isfile(path + '/' + 'checkpoint'):
    with open(path + '/' + 'checkpoint', 'r') as z:
        path_latest = z.readline()
        path_latest = path_latest.strip().split(' ')[-1]
        path_latest = path_latest.strip('"')
        path_latest = path + '/' + path_latest
        print(path_latest)

        os.system('zip t2t_' + args.name + ' ' + path_latest + '*')
        os.system('zip t2t_' + args.name + ' ' + path + '/*txt' + ' ' + path + '/*.json')

        os.system('mv t2t_' + args.name + '.zip' +  ' ..')