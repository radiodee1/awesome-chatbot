#!/usr/bin/env python3

import argparse
import os
import glob
import sys
sys.path.append('model/.')
from model.tokenize_weak import format

def is_good(line):
    #if line[0].isspace():
    #    return False
    line = line.strip()
    if len(line) == 0:
        return False
    if line[0].isdigit():
        return False

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make tab file from srt files.')
    parser.add_argument('--unzip' , help='subtitle files. (place zip files in "raw/srt_zip/" folder)' , action='store_true')
    parser.add_argument('--use-once', help='do not use each line as question and answer.', action='store_true')
    parser.add_argument('--flat', help='make flat output (no tabs and one sentence per line)', action='store_true')
    args = parser.parse_args()
    args = vars(args)

    if args['unzip'] is True:
        os.system('cd ./raw/srt_zip/. ; unzip -o "*.zip"')
        os.system('cd ./raw/srt_zip/. ; mv *.srt ..')

    filename = 'raw/*.srt'
    flag_use_both = not args['use_once']

    flag_make_flat = args['flat']

    filelist = []
    for i in glob.glob(filename):
        filelist.append(i)

    start_a = ''
    start_b = ''
    start_c = ''
    start_flat = ''
    print_out = False
    l_out = ''

    tot = 0
    with open('movie_srt_text.txt','w') as z:
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
                            if not start_flat.endswith(start_c):
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
                                    z.write(start_c + '\t' + start_b + '\t' + str(1) + '\n')
                                    tot += 1
                                elif flag_make_flat and len(start_flat.strip()) > 0 :
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
    pass