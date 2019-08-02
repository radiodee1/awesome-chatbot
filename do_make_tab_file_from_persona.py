#!/usr/bin/python3

import json
import re
import argparse
import os

personachat_file = './raw/personachat_self_original.json'
personachat_tab_file = './data/raw.txt'
data_dir = './data/'
end_of_list = '<-- end'

parser = argparse.ArgumentParser(
    description='Manipulate the persona dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--filter_negative', action='store_true', help='filter out words like *dont*')
parser.add_argument('--print', action='store_true', help='print to screen')
parser.add_argument('--separate', action='store_true',  help='make separate to/from files')
parser.add_argument('--zip', help='name for optional zip file.')
args = parser.parse_args()

filter_dont = args.filter_negative
print_to_screen = args.print
collect_history = True
multiple_files = args.separate

if args.zip is not None:
    if not args.zip.endswith('.zip'):
        args.zip += '.zip'

with open(personachat_file, "r", encoding="utf-8") as f:
    dataset = json.loads(f.read())


# Tokenize and encode the dataset using our loaded GPT tokenizer
def tokenize(obj, handle, space=' ', label='str:', write=False, recent_label=''):

    if write and print_to_screen and False:
        print('n == history')
    if isinstance(obj, str):
        if print_to_screen: print(space, len(space), label, obj)

        l = len(re.sub(' don','', obj))
        if len(obj) != l:
            print(obj)

        if (not (obj.startswith('__') or obj.endswith('__'))) and (not filter_dont or l == len(obj)):
            if (recent_label == 'history' or not collect_history) and obj != end_of_list:
                handle.write(obj + '\n')
        else:
            if print_to_screen: print(space, 'SILENCE tag')
        return obj

    if isinstance(obj, dict):
        if print_to_screen: print('--')
        if print_to_screen: print([n + ' ' + str(len(o))  for n ,o in obj.items() ])
        #if collect_history and n == 'history':

        return dict((n, tokenize(o, handle, space + ' ', label='dict:', write=(n == 'history'), recent_label=n)) for n, o in obj.items())

    if print_to_screen or True:
        return list(tokenize(o, handle, space + ' ', label='list:', write=(not collect_history or write), recent_label=recent_label) for o in obj + [end_of_list])


with open(personachat_tab_file,'w') as z:

    dataset2 = tokenize(dataset, z)

print('---')

file_list = []

if multiple_files:
    with open(personachat_tab_file, 'r') as z:
        zz = z.readlines()
        tr_fr = open(data_dir + 'train.big.from', 'w')
        tr_to = open(data_dir + 'train.big.to', 'w')
        file_list.append('train.big.from')
        file_list.append('train.big.to')
        for i in range(len(zz)):
            if i < len(zz) - 1:
                tr_fr.write(zz[i])
                tr_to.write(zz[i+1])
        tr_fr.close()
        tr_to.close()

        val_fr = open(data_dir + 'valid.big.from', 'w')
        val_to = open(data_dir + 'valid.big.to', 'w')
        file_list.append('valid.big.from')
        file_list.append('valid.big.to')
        val_num = int(len(zz) // 20)
        for i in range(len(zz) - val_num * 2 , len(zz) - val_num):
            if i < len(zz) - 1:
                val_fr.write(zz[i])
                val_to.write(zz[i+1])
        val_fr.close()
        val_to.close()

        test_fr = open(data_dir + 'test.big.from','w')
        test_to = open(data_dir + 'test.big.to','w')
        file_list.append('test.big.from')
        file_list.append('test.big.to')
        for i in range(len(zz) - val_num, len(zz)):
            if i < len(zz) - 1:
                test_fr.write(zz[i])
                test_to.write(zz[i+1])
        test_fr.close()
        test_to.close()

    if args.zip is not None:
        print(file_list)
        os.chdir(data_dir)
        os.system('zip ' + args.zip + ' ' + ' '.join(file_list))
        pass
    pass