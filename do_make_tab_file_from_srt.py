#!/usr/bin/env python3

#import random
import argparse
import os
import glob
import sys
sys.path.append('model/.')
from model.tokenize_weak import format

number = 0

def is_good(line):
    #if line[0].isspace():
    #    return False
    line = line.strip()
    if len(line) == 0:
        return False
    if line[0].isdigit():
        return False

    return True

def is_writable(line, keep=0.5, count=False):
    global number
    line = line.strip()
    if len(line.split(' ')) <= 1:
        return False
    if line.endswith('?'):
        return False
    if '\t' in line:
        line = line.split('\t')[0]
        if line.endswith('?'):
            return True
        if line[-1] in ',:;':
            return False
    elif '?' in line: ## questions
        return True
    if number < 0:
        return True
    if count:
        number += 1 ## statemnts
    if  (number % 5 ) * 20 > keep * 100:
        #print(line, number)
        return False
    return True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make tab file from srt files.')
    parser.add_argument('--unzip' , help='subtitle files. (place zip files in "raw/srt_zip/" folder)' , action='store_true')
    parser.add_argument('--use-once', help='do not use each line as question and answer.', action='store_true')
    parser.add_argument('--flat', help='make flat output (no tabs and one sentence per line - will not work with "--separate")', action='store_true')
    parser.add_argument('--separate', help='make separate input files also. (train.fr and train.to - will not work with "--flat")', action='store_true')
    parser.add_argument('--keep-statements', help='out of every 100 statements, what percentage to keep.', default=0.5)
    args = parser.parse_args()
    args = vars(args)

    if args['unzip'] is True:
        os.system('cd ./raw/srt_zip/. ; unzip -o "*.zip"')
        os.system('cd ./raw/srt_zip/. ; mv *.srt ..')

    filename = 'raw/*.srt'
    flag_use_both = not args['use_once']

    flag_make_flat = args['flat']
    flag_make_separate = args['separate']

    filelist = []
    for i in glob.glob(filename):
        filelist.append(i)

    start_a = ''
    start_b = ''
    start_c = ''
    start_flat = ''
    print_out = False
    l_out = ''

    filename_output = 'train_movie_srt.txt'
    keep = float(args['keep_statements'])

    tot = 0
    with open(filename_output,'w') as z:
        for i in filelist:
            if not os.path.isfile(i):
                print('bad file, ', i)
                continue
            with open(i, 'r') as r:
                lines = r.readlines()
                num = 0
                print_num = 3
                last_print_bad = num

                for l in lines:

                    if is_good(l):
                        l_out += ' ' + l.strip()

                        last_print_bad = num + 1
                        
                        if flag_make_flat:
                            if not start_flat.endswith(start_c.strip()):

                                start_flat += ' ' + start_c.strip()

                    if not is_good(l):

                        last_print_bad += 1
                        print_num += 1

                        if flag_use_both :

                            print_out = True
                        else:

                            if print_num % 2 == 0 or print_num < 6:
                                print_out = True


                        if print_out:

                            if (len(start_c) > 0 and len(start_b) > 0) and (len(l_out) > 0 or not flag_use_both):
                                if not flag_make_flat:
                                    if is_writable(start_c + '\t' + start_b, keep, count=True):
                                        z.write(start_c + '\t' + start_b + '\t' + str(1) + '\n')
                                        tot += 1
                                elif flag_make_flat and len(start_flat.strip()) > 1 :
                                    if is_writable(start_flat.strip() , keep, count=False): # and is_writable(start_c + '\t' + start_b, keep):
                                        number += 1
                                        z.write(start_flat.strip() + '\n')
                                        tot += 1

                            if not flag_use_both:
                                start_c = start_b
                                start_b = format(l_out)
                                pass

                            if len(l_out) > 0 and flag_use_both:
                                start_c = start_b
                                start_b = start_a
                                start_a = format(l_out)


                            if last_print_bad != num :
                                l_out = ''

                            print_out = False

                        if flag_make_flat:
                            start_flat = ''

                num += 1
            r.close()
    z.close()
    print('tot:', tot)

    if flag_make_separate and not flag_make_flat:
        with open(filename_output, 'r') as f, open('train.srt.fr.txt', 'w') as fr, open('train.srt.to.txt', 'w') as to:
            lines = f.readlines()
            for l in lines:
                l_from = l.split('\t')[0]
                l_to = l.split('\t')[1]
                fr.write(l_from + '\n')
                to.write(l_to + '\n')
        print('separate files written.')
        pass
    print('you may need to rename your files.')
    pass