#!/usr/bin/python3.6


import os as os
import sys
import argparse
from settings import hparams
import tokenize_weak

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

directions = {
    'n': 'north',
    's': 'south',
    'w': 'west',
    'e': 'east'
}


def format(s, split_phrases=False, add_sol_eol=False, add_eol_only=False, only_one_phrase=False):
    z = tokenize_weak.format(s)

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
                    if i.split(' ')[-1] != hparams['eol']:
                        x.append( i + ' ' + hparams['eol'] + ' . ')

        if only_one_phrase and len(x) > 1: ## return just first phrase
            return x[-1]
        x = ' '.join(x)
        return x
    if add_sol_eol:
        if not add_eol_only: z = hparams['sol'] + ' ' + z
        z = z + ' ' + hparams['eol']
    return z



def init_babi(fname, add_eol=False, replace_directions=False):
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

            if len(task["A"].split(',')) > 1 or add_eol:
                task["A"] = " ".join(task["A"].split(',')) + ' ' + hparams['eol']
                if replace_directions:
                    d = task["A"].split(' ')
                    dl = []
                    for a in d:
                        if a in directions:
                            a = directions[a]
                        dl.append(a)
                    task["A"] = ' '.join(dl)


                if add_eol and False:
                    task["A"] = task["A"] + ' . '

            if add_eol:
                #print('---')
                task["C"] = format(task["C"],split_phrases=True, add_eol_only=add_eol)
                task["Q"] = format(task["Q"],split_phrases=True, add_eol_only=add_eol)

            tasks.append(task.copy())

    return tasks


def get_babi_raw(id, test_id, sub_folder='en', add_eol=False, replace_directions=False):
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
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../raw/tasks_1-20_v1-2/%s/%s_train.txt' % (sub_folder, babi_name)), add_eol=add_eol, replace_directions=replace_directions)
    babi_test_raw = init_babi(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../raw/tasks_1-20_v1-2/%s/%s_test.txt' % (sub_folder, babi_test_name)), add_eol=add_eol, replace_directions=replace_directions)
    return babi_train_raw, babi_test_raw

if __name__ == '__main__':
    print('do make train and test')

    print('usage:')
    print(sys.argv[0], 'NUM , [folder]')
    print('NUM is either a number between 1 and 20 or the keyword "all". ')
    print('"all" produces a set of files with all 20 tests, in original order.')
    print('"folder" is either "en" or "en-10k"')

    parser = argparse.ArgumentParser(description='Make text files.')
    parser.add_argument('num', help='babi number, or "all"')
    parser.add_argument('folder', help='either "en-10k" or "en"')
    parser.add_argument('--eol', help='add eol to sentences', action='store_true')
    parser.add_argument('--replace-directions', help='replace directions with whole word.', action='store_true')
    args = parser.parse_args()
    args = vars(args)
    print(args)


    id = '1'
    sub_folder = 'en'
    flag_eol = False

    if len(sys.argv) > 1:
        #id = sys.argv[1]
        id = str(args['num'])
    print(id)
    if id == 'all': id_lst = [ str(i+1) for i in range(20)]
    else: id_lst = [id]
    if len(sys.argv) > 2:
        sub_folder = sys.argv[2] # 'en-10k'
        sub_folder = str(args['folder'])
    print(id_lst)
    if args['eol'] is True:
        flag_eol = True

    if flag_eol is False:
        hparams['eol'] = ''
        hparams['sol'] = ''

    mode = 'w'

    for id in id_lst:

        train, test = get_babi_raw(id,id, sub_folder=sub_folder, add_eol=flag_eol, replace_directions=args['replace_directions'])

        data_dir = hparams['data_dir']
        train_name = hparams['train_name']
        test_name = hparams['test_name']
        valid_name = hparams['valid_name']
        babi_name = hparams['babi_name']
        src_ending = hparams['src_ending']
        tgt_ending = hparams['tgt_ending']
        question_ending = hparams['question_ending']

        split = len(test) * 0.5 #/ 2
        print(split,'split')

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