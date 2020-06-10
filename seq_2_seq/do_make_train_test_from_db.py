#!/usr/bin/env python3


import sqlite3
import pandas as pd
import os
import argparse
import random
import re
import tokenize_weak
import sys
from settings import hparams
#import core.tokenizer as ct

timeframes = ['input']

print(sys.argv)

choices = [
    'hello',
    'goodbye',
    'what',
    'what is your name',
    'i agree'
]

parser = argparse.ArgumentParser(description='Make text files.')
parser.add_argument('--basefile', metavar='FILE', type=str, help='Base database file name')
parser.add_argument('--babi', help='save triplets in stead of pairs', action='store_true')
parser.add_argument('--autoencode', help='store auto encoder format', action='store_true')
parser.add_argument('--to-lower', help='store in lowercase form', action='store_true')
parser.add_argument('--test-on-screen', help='test on screen', action='store_true')
parser.add_argument('--subdevide', help='subdevide into batches', action='store_true')
parser.add_argument('--only-one',help='record only one phrase for each question/answer.', action='store_true')
parser.add_argument('--length', help='number of examples to process')
parser.add_argument('--to-gpt2', help='no special tokens (eol, sol), etc.', action='store_true')
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
do_gpt2 = False
do_autoencode = False
do_autoencode_context = False
do_autoencode_question = False
do_only_one_phrase = False
approximate_length = 0
count_recorded = 0

if args['babi'] is True: do_babi = True
if args['to_lower'] is True: to_lower = True
if args['test_on_screen'] is True: test_on_screen = True
if args['subdevide'] is True: subdevide_into_batches = True
if args['autoencode'] is True:
    do_autoencode = True
    ## set the context and question variables by hand
    do_autoencode_context = True
    do_autoencode_question = False
if args['length'] is not None:
    approximate_length = int(args['length'])
if args['only_one'] is not False: do_only_one_phrase = True

if args['to_gpt2'] is True:
    hparams['sol'] = ''
    hparams['eol'] = ''
    do_gpt2 = True

batch_size = 64 #32 # 64 #256
steps_per_stats = 100
#pull_size = batch_size * steps_per_stats * 10

test_name = hparams['test_name']
train_name = hparams['train_name']
valid_name = hparams['valid_name']

src_ending = hparams['src_ending']
tgt_ending = hparams['tgt_ending']
question_ending = hparams['question_ending']

def format(s, split_phrases=False, add_sol_eol=False, add_eol_only=False, only_one_phrase=False):
    z = tokenize_weak.format(s)
    if z == None or z.strip() == '':

        #z = ' hello '
        z = random.choice(choices)
        add_sol_eol = True
        add_eol_only = True

    if split_phrases:
        x = []

        z = z.replace(',', ' ')
        z = z.replace('?', ' ? . ')
        z = z.replace('!', ' . ')
        zz = z.split('.')
        #zz = filter(None, re.split("[,.\-!?:]+", z))
        for i in zz:
            xx = i.split(' ')
            y = []
            for j in xx:
                j = j.strip()
                if len(j) > 0: y.append(j)
            i = ' '.join(y)
            i = i.strip()
            if len(i) > 1 and not i.isspace():
                if not add_eol_only:
                    x.append( hparams['sol'] + ' ' + i + ' ' + hparams['eol'] + ' . ' )
                else:
                    x.append( i + ' ' + hparams['eol'] + ' . ')

        if only_one_phrase and len(x) > 1: ## return just first phrase
            return x[-1]
        x = ' '.join(x)
        return x
    if add_sol_eol:
        if not add_eol_only: z = hparams['sol'] + ' ' + z
        z = z + ' ' + hparams['eol']
    return z


try:
    for timeframe in timeframes:
        connection = sqlite3.connect('{}.db'.format(timeframe))
        c = connection.cursor()
        limit = 1000
        last_unix = 0
        cur_length = limit
        counter = 0
        counter_occur = 0
        test_done = False
        pull_num = 0
        mode = 'w'

        while cur_length == limit and (count_recorded < approximate_length + limit or approximate_length == 0):

            if args['to_gpt2'] is True:
                df = pd.read_sql("SELECT * FROM parent_reply WHERE  parent NOT NULL ORDER BY unix ASC LIMIT {}".format(limit),connection)
            else:
                df = pd.read_sql("SELECT * FROM parent_reply WHERE unix > {} and parent NOT NULL and score > 0 ORDER BY unix ASC LIMIT {}".format(last_unix,limit),connection)

            try:
                #print(df['unix'].values)
                last_unix =df.tail(1)['unix'].values[0]
                #print(last_unix)
                #last_unix = df['unix'].values[0]

            except:
                print('error')

                last_unix = 0

            cur_length = len(df)

            content_parent = df['parent'].values
            content_comment = df['comment'].values

            skip_num = 0

            src_list = []
            tgt_list = []
            ques_list = []

            if count_recorded + 1 >= approximate_length + limit and approximate_length != 0:
                print('skipping.')
                break

            if do_gpt2 and (count_recorded < approximate_length + limit or approximate_length == 0):
                tmp = ''
                for i in range(len(content_parent)):

                    tmp = format(content_parent[i])
                    tmp += '\t ' + format(content_comment[i])
                    #tmp = format(tmp , split_phrases=True, add_eol_only=True, only_one_phrase=do_only_one_phrase)

                    tmpz = tmp.split('.')

                    ## get last sentence ##
                    #if len(tmpz) > 1: tmpz = tmpz[-2]
                    #else: tmpz = ''

                    z_len_1 = len(content_parent[i].split(' '))
                    z_len_2 = len(content_comment[i].split(' '))


                    if not test_done or (len(tmp) > 0 and z_len_1 > 0 and z_len_2 > 0 and
                            (count_recorded < approximate_length + limit or approximate_length == 0)):

                        if True:
                            #tmp = tmpz # simplify autoencode situation
                            if tmp.strip() == '':
                                tmp = random.choice(choices)
                                print('empty string found.')

                        l_list = tmp.split()
                        if 'name' in l_list:
                            counter_occur += 1
                            #print(tmp)

                        src_list.append(tmp)
                        #else: src_list.append('')

                        #if do_autoencode_question:
                        ques_list.append(tmp)
                        #else: ques_list.append('')

                        tgt_list.append(tmp)
                        count_recorded += 1
                    elif count_recorded >= approximate_length + limit and approximate_length != 0:
                        print('last recorded.')
                        break
                    else:
                        skip_num += 1
                        print('skip one here!', skip_num)


                pass
            elif do_autoencode and do_babi and (count_recorded < approximate_length + limit or approximate_length == 0):
                tmp = ''
                assert do_autoencode_context is not do_autoencode_question

                for i in range(len(content_parent)):

                    tmp = content_parent[i]
                    tmp += ' . ' + content_comment[i]
                    tmp = format(tmp , split_phrases=True, add_eol_only=True, only_one_phrase=do_only_one_phrase)

                    tmpz = tmp.split('.')

                    ## get last sentence ##
                    if len(tmpz) > 1: tmpz = tmpz[-2]
                    else: tmpz = ''

                    z_len_1 = len(content_parent[i].split(' '))
                    z_len_2 = len(content_comment[i].split(' '))


                    if not test_done or (len(tmpz) > 0 and len(tmp) > 0 and z_len_1 > 0 and z_len_2 > 0 and
                            (count_recorded < approximate_length + limit or approximate_length == 0)):

                        if True:
                            tmp = tmpz # simplify autoencode situation
                            if tmp.strip() == '':
                                tmp = random.choice(choices)
                                print('empty string found.')

                        if do_autoencode_context: src_list.append(tmp)
                        else: src_list.append('')

                        if do_autoencode_question: ques_list.append(tmpz)
                        else: ques_list.append('')

                        tgt_list.append(tmpz)
                        count_recorded += 1
                    elif count_recorded >= approximate_length + limit and approximate_length != 0:
                        print('last recorded.')
                        break
                    else:
                        skip_num += 1
                        print('skip one here!', skip_num)

                pass
            elif  (count_recorded < approximate_length + limit or approximate_length == 0):

                for i in range(len(content_parent)):

                    src_list.append(format(content_parent[i],split_phrases=True, add_eol_only=True, only_one_phrase=do_only_one_phrase))
                    tgt_list.append(format(content_comment[i],split_phrases=True, add_eol_only=True, only_one_phrase=do_only_one_phrase))
                    if do_babi:
                        ques_list.append(format(content_parent[i],split_phrases=True, add_eol_only=True, only_one_phrase=do_only_one_phrase))

                    count_recorded += 1 #len(content_parent)

                    if count_recorded >= approximate_length + limit and approximate_length != 0:
                        print('last recorded.')
                        break

            ## record values ##

            if do_gpt2:
                split = 0 #len(src_list)

                with open('../raw/' + 'chat' + '_' + 'reddit_tab.txt', mode, encoding='utf8') as f:
                    for content in src_list[split:]:  # df['parent'].values:
                        #content = format(content) ## this removes tab chars!!
                        f.write(str(content) + '\n')

                mode = 'a'

            elif not test_done and (count_recorded < approximate_length + limit or approximate_length == 0):

                split = int(len(src_list) * 0.5)

                with open('../raw/' + test_name + '.'+ src_ending, mode, encoding='utf8') as f:
                    for content in src_list[split:]: # df['parent'].values:
                        content = format(content)
                        f.write(str(content)+'\n')

                with open('../raw/' + test_name + '.' + tgt_ending, mode, encoding='utf8') as f:
                    for content in tgt_list[split:]: #df['comment'].values:
                        content = format(content)
                        f.write(str(content)+'\n')

                if do_babi:
                    with open('../raw/' + test_name + '.' + question_ending, mode, encoding='utf8') as f:
                        for content in ques_list[split:]:  # df['comment'].values:
                            content = format(content)
                            f.write(str(content) + '\n')
                    pass


                ####################

                with open('../raw/' + valid_name + '.'+ src_ending, mode, encoding='utf8') as f:
                    for content in src_list[:split]: # df['parent'].values:
                        content = format(content)
                        f.write(str(content)+'\n')

                with open('../raw/' + valid_name + '.' + tgt_ending, mode, encoding='utf8') as f:
                    for content in tgt_list[:split]: #df['comment'].values:
                        content = format(content)
                        f.write(str(content)+'\n')

                if do_babi:
                    with open('../raw/' + valid_name + '.' + question_ending, mode, encoding='utf8') as f:
                        for content in ques_list[:split]:  # df['comment'].values:
                            content = format(content)
                            f.write(str(content) + '\n')
                    pass

                test_done = True
                #limit = 5000
                #limit = pull_size
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

                if not do_gpt2: mode = 'a'

            counter += 1
            if counter > 3 and test_on_screen: exit()
            if counter % limit == 0 or True:
                print(counter * limit, limit, counter, 'rows/iters completed so far', end=' ')
                if do_gpt2:
                    print('-' ,counter_occur, '"name" mentioned')
                else:
                    print()

except KeyboardInterrupt:
    print()
    pass
finally:

    if not test_on_screen  and not do_gpt2:
        s = 'mv ../raw/train* ../raw/test* ../raw/valid* ../data/.'
        print(s)
        os.system(s)
        pass
