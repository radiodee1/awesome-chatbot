#!/usr/bin/python3.6

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
    args = parser.parse_args()
    args = vars(args)

    if args['unzip'] is True:
        os.system('cd ./raw/srt_zip/. ; unzip -o "*.zip"')
        os.system('cd ./raw/srt_zip/. ; mv *.srt ..')

    filename = 'raw/*.srt'
    flag_use_both = True

    filelist = []
    for i in glob.glob(filename):
        filelist.append(i)

    start_a = ''
    start_b = ''
    print_out = False
    l_out = ''
    with open('movie_srt_text.txt','w') as z:
        for i in filelist:
            if not os.path.isfile(i):
                print('bad file, ', i)
                continue
            with open(i, 'r') as r:
                lines = r.readlines()
                for l in lines:

                    l_out += ' ' + l.strip()

                    #start_b = start_a

                    if not is_good(l):
                        if flag_use_both : #or print_out:

                            start_b = start_a
                            start_a = format(l_out)

                            if len(start_b) > 0 and len(start_a) > 0 :
                                z.write(start_a + '\t' + start_b + '\t' + str(1) + '\n')
                                #l_out = ''
                        else:
                            start_b = start_a
                            start_a = format(l_out)

                            print_out = True
                            #print(start_a, start_b)
                        l_out = ''
                    else:
                        pass #start_b = start_a

    pass