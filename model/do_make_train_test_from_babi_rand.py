#!/usr/bin/python3.6


import os as os
import sys
from settings import hparams

'''
The MIT License (MIT)

Copyright (c) 2016 YerevaNN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

def init_babi(fname):
    print("==> Loading file from %s" % fname)

    tasks = []
    task = None
    for i, line in enumerate(open(fname)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": ""}

        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ') + 1:]
        if line.find('?') == -1:
            task["C"] += line.lower()
        else:
            idx = line.find('?')
            tmp = line[idx + 1:].split('\t')
            task["Q"] = line[:idx].lower()
            task["A"] = tmp[1].strip().lower()
            tasks.append(task.copy())

    return tasks


def get_babi_raw(id, test_id):
    babi_map = {
        "1": "qa1_single-supporting-fact",
        "2": "qa2_two-supporting-facts",
        "3": "qa3_three-supporting-facts",
        "4": "qa4_two-arg-relations",
        "5": "qa5_three-arg-relations",
        "6": "qa6_yes-no-questions",
        "7": "qa7_counting",
        "8": "qa8_lists-sets",
        "9": "qa9_simple-negation",
        "10": "qa10_indefinite-knowledge",
        "11": "qa11_basic-coreference",
        "12": "qa12_conjunction",
        "13": "qa13_compound-coreference",
        "14": "qa14_time-reasoning",
        "15": "qa15_basic-deduction",
        "16": "qa16_basic-induction",
        "17": "qa17_positional-reasoning",
        "18": "qa18_size-reasoning",
        "19": "qa19_path-finding",
        "20": "qa20_agents-motivations",
        "MCTest": "MCTest",
        "19changed": "19changed",
        "joint": "all_shuffled",
        "sh1": "../shuffled/qa1_single-supporting-fact",
        "sh2": "../shuffled/qa2_two-supporting-facts",
        "sh3": "../shuffled/qa3_three-supporting-facts",
        "sh4": "../shuffled/qa4_two-arg-relations",
        "sh5": "../shuffled/qa5_three-arg-relations",
        "sh6": "../shuffled/qa6_yes-no-questions",
        "sh7": "../shuffled/qa7_counting",
        "sh8": "../shuffled/qa8_lists-sets",
        "sh9": "../shuffled/qa9_simple-negation",
        "sh10": "../shuffled/qa10_indefinite-knowledge",
        "sh11": "../shuffled/qa11_basic-coreference",
        "sh12": "../shuffled/qa12_conjunction",
        "sh13": "../shuffled/qa13_compound-coreference",
        "sh14": "../shuffled/qa14_time-reasoning",
        "sh15": "../shuffled/qa15_basic-deduction",
        "sh16": "../shuffled/qa16_basic-induction",
        "sh17": "../shuffled/qa17_positional-reasoning",
        "sh18": "../shuffled/qa18_size-reasoning",
        "sh19": "../shuffled/qa19_path-finding",
        "sh20": "../shuffled/qa20_agents-motivations",
    }
    if (test_id == ""):
        test_id = id
    babi_name = babi_map[id]
    babi_test_name = babi_map[test_id]
    babi_train_raw = init_babi(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../raw/tasks_1-20_v1-2/en/%s_train.txt' % babi_name))
    babi_test_raw = init_babi(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../raw/tasks_1-20_v1-2/en/%s_test.txt' % babi_test_name))
    return babi_train_raw, babi_test_raw

if __name__ == '__main__':
    print('do make train and test')

    print('usage:')
    print(sys.argv[0], 'NUM')
    print('NUM is either a number between 1 and 20 or the keyword "all". ')
    print('"all" produces a set of files with all 20 tests, in original order.')

    id = '1'
    if len(sys.argv) > 1:
        id = sys.argv[1]
    print(id)
    if id == 'all': id_lst = [ str(i+1) for i in range(20)]
    else: id_lst = [id]
    print(id_lst)


    mode = 'w'

    for id in id_lst:

        train, test = get_babi_raw(id,id)

        data_dir = hparams['data_dir']
        train_name = hparams['train_name']
        test_name = hparams['test_name']
        valid_name = hparams['valid_name']
        babi_name = hparams['babi_name']
        src_ending = hparams['src_ending']
        tgt_ending = hparams['tgt_ending']
        question_ending = hparams['question_ending']

        split = len(test) / 2

        with open(data_dir + train_name +'.'+ babi_name + '.' + src_ending, mode) as z:
            for i in range(len(train)):
                z.write(train[i]['C'] + '\n')
                pass

        with open(data_dir + train_name +'.'+ babi_name + '.' + tgt_ending, mode) as z:
            for i in range(len(train)):
                z.write(train[i]['A'] + '\n')
                pass

        with open(data_dir + train_name +'.'+ babi_name + '.' + question_ending, mode) as z:
            for i in range(len(train)):
                z.write(train[i]['Q'] + '\n')
                pass

        ######################
        with open(data_dir + test_name + '.' + babi_name + '.' + src_ending, mode) as z:
            for i in range(len(test)):
                if i < split:
                    z.write(test[i]['C'] + '\n')
                pass

        with open(data_dir + test_name + '.' + babi_name + '.' + tgt_ending, mode) as z:
            for i in range(len(test)):
                if i < split:
                    z.write(test[i]['A'] + '\n')
                pass

        with open(data_dir + test_name + '.' + babi_name + '.' + question_ending, mode) as z:
            for i in range(len(test)):
                if i < split:
                    z.write(test[i]['Q'] + '\n')
                pass

        ######################
        with open(data_dir + valid_name + '.' + babi_name + '.' + src_ending, mode) as z:
            for i in range(len(test)):
                if i >= split:
                    z.write(test[i]['C'] + '\n')
                pass

        with open(data_dir + valid_name + '.' + babi_name + '.' + tgt_ending, mode) as z:
            for i in range(len(test)):
                if i >= split:
                    z.write(test[i]['A'] + '\n')
                pass

        with open(data_dir + valid_name + '.' + babi_name + '.' + question_ending, mode) as z:
            for i in range(len(test)):
                if i >= split:
                    z.write(test[i]['Q'] + '\n')
                pass

        if len(id_lst) > 1: mode = 'a'