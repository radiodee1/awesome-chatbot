#!/usr/bin/python3


import sqlite3
import pandas as pd
import os
import argparse
import tokenize_weak
import sys
from settings import hparams
#import core.tokenizer as ct

timeframes = ['input']

print(sys.argv)

parser = argparse.ArgumentParser(description='Make text files.')
parser.add_argument('--basefile', metavar='FILE', type=str, help='Base database file name')
parser.add_argument('--babi', help='save triplets in stead of pairs', action='store_true')
parser.add_argument('--autoencode', help='store auto encoder format', action='store_true')
parser.add_argument('--to-lower', help='store in lowercase form', action='store_true')
parser.add_argument('--test-on-screen', help='test on screen', action='store_true')
parser.add_argument('--subdevide', help='subdevide into batches', action='store_true')
args = parser.parse_args()
args = vars(args)
print(args)

if args['basefile'] is not None: #len(sys.argv) > 1:
    z = args['basefile'].split('.')
    z = '.'.join(z[:-1])
    timeframes = [str(z)]

to_lower = False
test_on_screen = False
subdevide_into_batches = False
do_babi = False
do_autoencode = False
do_autoencode_context = False
do_autoencode_question = False

if args['babi'] is True: do_babi = True
if args['to_lower'] is True: to_lower = True
if args['test_on_screen'] is True: test_on_screen = True
if args['subdevide'] is True: subdevide_into_batches = True
if args['autoencode'] is True:
    do_autoencode = True
    ## set the context and question variables by hand
    do_autoencode_context = True
    do_autoencode_question = False

batch_size = 64 #32 # 64 #256
steps_per_stats = 100
pull_size = batch_size * steps_per_stats * 10

test_name = hparams['test_name']
train_name = hparams['train_name']
valid_name = hparams['valid_name']

src_ending = hparams['src_ending']
tgt_ending = hparams['tgt_ending']
question_ending = hparams['question_ending']

def format(s, split_phrases=False, add_sol_eol=False):
    z = tokenize_weak.format(s)
    if z == None or z.strip() == '':
        z = ' what ? '
        add_sol_eol = True

    if split_phrases:
        x = []
        z = z.replace(',', ' . ')
        z = z.replace('?', ' . ')
        z = z.replace('!', ' . ')
        zz = z.split('.')
        for i in zz:
            i = i.strip()
            if len(i) > 1:
                x.append( hparams['sol'] + ' ' + i + ' ' + hparams['eol'] + ' . ' )
        x = ' '.join(x)
        return x
    if add_sol_eol:
        z = hparams['sol'] + ' ' + z
        z = z + ' ' + hparams['eol']
    return z

try:
    for timeframe in timeframes:
        connection = sqlite3.connect('{}.db'.format(timeframe))
        c = connection.cursor()
        limit = 100 #5000
        last_unix = 0
        cur_length = limit
        counter = 0
        test_done = False
        pull_num = 0
        mode = 'w'

        while cur_length == limit:

            df = pd.read_sql("SELECT * FROM parent_reply WHERE unix > {} and parent NOT NULL and score > 0 ORDER BY unix ASC LIMIT {}".format(last_unix,limit),connection)

            try:
                last_unix = df.tail(1)['unix'].values[0]
            except:
                print('error')

                last_unix = 0

            cur_length = len(df)

            content_parent = df['parent'].values
            content_comment = df['comment'].values


            src_list = []
            tgt_list = []
            ques_list = []

            if do_autoencode and do_babi:
                tmp = ''
                assert do_autoencode_context is not do_autoencode_question

                for i in range(len(content_parent)):

                    tmp = content_parent[i]
                    tmp += ' . ' + content_comment[i]
                    tmp = format(tmp , split_phrases=True)

                    tmpz = tmp.split('.')

                    ## get last sentence ##
                    if len(tmpz) > 1: tmpz = tmpz[-2]
                    else: tmpz = ''


                    if len(tmpz) > 0 and len(tmp) > 0:
                        if do_autoencode_context: src_list.append(tmp)
                        else: src_list.append('')

                        if do_autoencode_question: ques_list.append(tmpz)
                        else: ques_list.append('')

                        tgt_list.append(tmpz)

                pass
            else:

                src_list = content_parent[:]
                tgt_list = content_comment[:]



            if not test_done:

                with open('../raw/' + test_name + '.'+ src_ending, mode, encoding='utf8') as f:
                    for content in src_list: # df['parent'].values:
                        content = format(content)
                        f.write(str(content)+'\n')

                with open('../raw/' + test_name + '.' + tgt_ending, mode, encoding='utf8') as f:
                    for content in tgt_list: #df['comment'].values:
                        content = format(content)
                        f.write(str(content)+'\n')

                if do_babi:
                    with open('../raw/' + test_name + '.' + question_ending, mode, encoding='utf8') as f:
                        for content in ques_list:  # df['comment'].values:
                            content = format(content)
                            f.write(str(content) + '\n')
                    pass

                test_done = True
                #limit = 5000
                limit = pull_size
                cur_length = limit

            else:

                with open('../raw/' + train_name + '.big.' + src_ending, mode, encoding='utf8') as f:
                    for content in src_list: #df['parent'].values:
                        content = format(content)
                        f.write(str(content)+'\n')

                with open('../raw/' + train_name + '.big.' + tgt_ending, mode, encoding='utf8') as f:
                    for content in tgt_list:

                        content = format(content)
                        f.write(str(content)+'\n')

                if do_babi:
                    with open('../raw/' + train_name + '.big.' + question_ending, mode, encoding='utf8') as f:
                        for content in ques_list:  # df['comment'].values:
                            content = format(content)
                            f.write(str(content) + '\n')
                    pass

                pull_num += 1
                if subdevide_into_batches:
                    with open('../raw/' + train_name + '.' + str(pull_num) + '.' + src_ending, mode, encoding='utf8') as f:
                        for content in src_list: #df['parent'].values:
                            content = format(content)
                            f.write(str(content)+'\n')

                    with open('../raw/' + train_name + '.' + str(pull_num) + '.' + tgt_ending, mode, encoding='utf8') as f:
                        for content in tgt_list: #df['comment'].values:
                            content = format(content)
                            f.write(str(content)+'\n')

                    if do_babi:
                        with open('../raw/' + train_name + '.' + question_ending, mode, encoding='utf8') as f:
                            for content in ques_list:  # df['comment'].values:
                                content = format(content)
                                f.write(str(content) + '\n')
                        pass

                mode = 'a'

            counter += 1
            if counter > 3 and test_on_screen: exit()
            if counter % pull_size == 0 or True:
                print(counter * limit, counter, 'rows/iters completed so far')

except KeyboardInterrupt:

    if not test_on_screen:
        os.system('mv ../raw/train* ../raw/test* ../data/.')
        pass
