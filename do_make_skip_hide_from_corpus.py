#!/usr/bin/python3

import argparse
import os
import sys
import re

print('''
For skip and hide enter list in the form of a comma separated string. An example is:
--skip="don,no" --hide="sol,eol,unk"

'Hide' options remove a word from a sentence. 'Skip' options remove a sentence pair from
the list of all sentences.
''')

parser = argparse.ArgumentParser()
parser.add_argument('--filename', default='./data/train.big.from', help='name of input file. (DEFAULT: ./data/train.big.from)')
parser.add_argument('--skip', default='don,no', help='comma separated list of words to skip.')
parser.add_argument('--hide', default='sol,eol,unk', help='comma separated list of words to hide.')
parser.add_argument('--zip', help='name of optional zip file for archive.')

args = parser.parse_args()

print(args)

skip_list = args.skip.split(',')
hide_list = args.hide.split(',')

dir_start = os.path.dirname(os.path.realpath(__file__))

def skip_hide(filename, skip_list, hide_list):
    dir = filename.split('/')[:-1]
    dir = '/'.join(dir) + '/'
    basename = filename.split('/')[-1]
    basename = basename.split('.')[:-1]
    basename = '.'.join(basename)

    name_from = dir + basename + '.from'
    name_to = dir + basename + '.to'
    name_from_tmp = name_from + '.tmp'
    name_to_tmp = name_to + '.tmp'

    print(basename)

    if os.path.isfile(name_from) and os.path.isfile(name_to):
        tmp_count = 0

        file_from = open(name_from, 'r')
        file_to = open(name_to, 'r')
        file_from_tmp = open(name_from_tmp,'w')
        file_to_tmp = open(name_to_tmp,'w')
        list_from = file_from.readlines()
        list_to = file_to.readlines()
        count = len(list_from)
        if len(list_from) == len(list_to):
            for i in range(len(list_from)):
                do_write = True
                tmp_sentence_from = list_from[i]
                tmp_sentence_to = list_to[i]
                for j in hide_list:
                    tmp_sentence_from = tmp_sentence_from.replace(j ,'')
                    tmp_sentence_to = tmp_sentence_to.replace(j, '')

                for j in skip_list:
                    if len(tmp_sentence_from) != len(tmp_sentence_from.replace(j,'')):
                        do_write = False
                    if len(tmp_sentence_to) != len(tmp_sentence_to.replace(j, '')):
                        do_write = False
                if do_write:
                    file_from_tmp.write(tmp_sentence_from)
                    file_to_tmp.write(tmp_sentence_to)
                    tmp_count +=1

            print('filesize before:\t', name_from, count)
            print('filesize after: \t', name_from_tmp, tmp_count)
        else:
            print('files not equal.')

        file_from.close()
        file_to.close()
        file_from_tmp.close()
        file_to_tmp.close()
        if tmp_count > 0 and count > tmp_count:
            print('replace original with modified.')
            os.system('rm ' + name_from + ' ' + name_to)
            os.system('mv ' + name_from_tmp + ' ' + name_from)
            os.system('mv ' + name_to_tmp + ' ' + name_to)

        else:
            print('no change! NO ZIP')
            os.system('rm ' + name_from_tmp + ' ' + name_to_tmp)
    print(name_from, name_to, name_from_tmp, name_to_tmp)
    if args.zip is not None:
        if not args.zip.endswith('.zip'):
            args.zip += '.zip'
        os.chdir(dir)
        print(name_from[len(dir):], name_to[len(dir):])
        os.system('zip ' + args.zip + ' ' + name_from[len(dir):] + ' ' + name_to[len(dir):])
        os.chdir(dir_start)
    pass

if __name__ == '__main__':
    print('----')
    skip_hide(args.filename, skip_list, hide_list)
    filename = args.filename.replace('train', 'test')
    print(filename,'test ----')
    skip_hide(filename, skip_list, hide_list)
    filename = args.filename.replace('train', 'valid')
    print(filename, 'valid ----')
    skip_hide(filename, skip_list, hide_list)