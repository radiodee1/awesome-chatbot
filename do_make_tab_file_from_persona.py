#!/usr/bin/python3

import json
import re

personachat_file = './raw/personachat_self_original.json'
personachat_tab_file = './data/raw.txt'
end_of_list = '<-- end'

filter_dont = True
print_to_screen = False
collect_history = True

with open(personachat_file, "r", encoding="utf-8") as f:
    dataset = json.loads(f.read())


# Tokenize and encode the dataset using our loaded GPT tokenizer
def tokenize(obj, handle, space=' ', label='str:', write=False, recent_label=''):

    if write and print_to_screen and False:
        print('n == history')
    if isinstance(obj, str):
        if print_to_screen: print(space, len(space), label, obj)

        l = len(re.sub(' don','', obj))
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
