#!/usr/bin/python3

import os
import argparse
import sys
import subprocess

print('''
usage: use bash completion with the basename option. If you denote a checkpoint file, the
program will save that checkpoint. If you denote an export file, the program will try to
save all files associated with that export.
''')

parser = argparse.ArgumentParser(
    description='zip tensor-2-tensor on your chat dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--basename', nargs='*', help='start zip operation on this and other files.')
parser.add_argument('--latest', action='store_true', help='choose latest checkpoint.')
parser.add_argument('--name', default='chat', help='run filename.')
parser.add_argument('--no-vocab', action='store_true', help='skip storing vocab files.')
args = parser.parse_args()

if args.basename is not None :
    args.basename = args.basename[0]
    print(args.basename)

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
        name = args.name
        args.name += '.exported.zip'
        print(args.name)
        path_export = args.basename.strip().split('/')
        #exit('bad model name')
        if os.path.isdir('/'.join(path_export)):
            basename = '/'.join(path_export)
            if not basename.endswith('/'):
                basename = basename + '/'
            basename = basename.strip().split('/')[-2]

        print(basename)
        if basename.startswith('saved_model') or basename.startswith('variables') or basename.isdigit():
            print('server?')
            while (path_export[-1] != 'variables' and
                   not path_export[-1].startswith('saved') and
                   not path_export[-1].strip('/').isdigit()) and len(path_export) > 2:
                path_export = path_export[:-1]
            if not path_export[-1].isdigit():
                path_export = path_export[:-1]

            #print('server vars', path_export)
            path_export = '/'.join(path_export)
            print(path_export)
            subprocess.call('zip -r t2t_' + args.name + ' ' + path_export + '*', shell=True)
            #os.system('mv t2t_' + args.name + '.zip  ./..')
            #name = args.name
            vocabname = 'data/t2t_data/' + name + '/' + 'vocab.' + '*'

            if not args.no_vocab:
                subprocess.call('zip -r t2t_' + args.name + ' ' + vocabname, shell=True)

            subprocess.call('mv t2t_' + args.name  + ' ..', shell=True)
            exit()

print(path)
name = args.name

if os.path.isfile(path + '/' + 'checkpoint'):
    with open(path + '/' + 'checkpoint', 'r') as z:
        name = args.name + '.checkpoint.zip'
        path_latest = z.readline()
        path_latest = path_latest.strip().split(' ')[-1]
        path_latest = path_latest.strip('"')
        path_latest = path + '/' + path_latest
        print(path_latest)

        subprocess.call('zip -r t2t_' + name + ' ' + path_latest + '*', shell=True)
        subprocess.call('zip -r t2t_' + name + ' ' + path + '/*txt' + ' ' + path + '/*.json', shell=True)



        vocabname = 'data/t2t_data/' + args.name + '/' + 'vocab.' + '*'

        if not args.no_vocab:
            subprocess.call('zip -r t2t_' + name + ' ' + vocabname, shell=True)

        subprocess.call('mv t2t_' + name  +  ' ..', shell=True)