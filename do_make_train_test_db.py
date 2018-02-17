#!/usr/bin/python3

import sqlite3
import pandas as pd
import os
import tokenize_weak
import sys
#import core.tokenizer as ct

timeframes = ['input']

print(sys.argv)

if len(sys.argv) > 1:
    z = sys.argv[1].split('.')
    z = '.'.join(z[:-1])
    timeframes = [str(z)]

to_lower = True
test_on_screen = False
remove_caps = True

batch_size = 256
steps_per_stats = 100
pull_size = batch_size * steps_per_stats * 10

def format(s):
    z = tokenize_weak.format(s)
    if z.strip() == '':
        z = ' what ? '
    return z

for timeframe in timeframes:
    connection = sqlite3.connect('{}.db'.format(timeframe))
    c = connection.cursor()
    limit = 100 #5000
    last_unix = 0
    cur_length = limit
    counter = 0
    test_done = False
    pull_num = 0

    while cur_length == limit:

        df = pd.read_sql("SELECT * FROM parent_reply WHERE unix > {} and parent NOT NULL and score > 0 ORDER BY unix ASC LIMIT {}".format(last_unix,limit),connection)

        try:
            last_unix = df.tail(1)['unix'].values[0]
        except:
            print('error')

            last_unix = 0

        cur_length = len(df)

        if not test_done:
            with open('raw/test.from','a', encoding='utf8') as f:
                for content in df['parent'].values:
                    content = format(content)
                    f.write(str(content)+'\n')

            with open('raw/test.to','a', encoding='utf8') as f:
                for content in df['comment'].values:
                    content = format(content)
                    f.write(str(content)+'\n')

            test_done = True
            #limit = 5000
            limit = pull_size
            cur_length = limit

        else:

            with open('raw/train.big.from','a', encoding='utf8') as f:
                for content in df['parent'].values:
                    content = format(content)
                    f.write(str(content)+'\n')

            with open('raw/train.big.to','a', encoding='utf8') as f:
                for content in df['comment'].values:
                    content = format(content)
                    f.write(str(content)+'\n')

            pull_num += 1
            with open('raw/train.'+ str(pull_num) + '.from','a', encoding='utf8') as f:
                for content in df['parent'].values:
                    content = format(content)
                    f.write(str(content)+'\n')

            with open('raw/train.'+ str(pull_num) + '.to','a', encoding='utf8') as f:
                for content in df['comment'].values:
                    content = format(content)
                    f.write(str(content)+'\n')


        counter += 1
        if counter > 3 and test_on_screen: exit()
        if counter % pull_size == 0 or True:
            print(counter * limit, counter, 'rows completed so far')
            
    if not test_on_screen:
        #os.system('mv train.big.from train.big.to test.from test.to new_data/.')
        os.system('mv raw/t* new_data/.')
