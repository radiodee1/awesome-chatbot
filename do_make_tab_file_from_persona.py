#!/usr/bin/python3

import json
import re

personachat_file = './raw/personachat_self_original.json'

personachat_tab_file = './data/raw.txt'

filter_dont = True
print_to_screen = False

with open(personachat_file, "r", encoding="utf-8") as f:
    dataset = json.loads(f.read())


# Tokenize and encode the dataset using our loaded GPT tokenizer
def tokenize(obj, handle, space=' '):

    if isinstance(obj, str):
        if print_to_screen: print(space, len(space), obj)
        l = len(re.sub(' don','', obj))
        if (not (obj.startswith('__') or obj.endswith('__'))) and (not filter_dont or l == len(obj)):
            handle.write(obj + '\n')
        else:
            if print_to_screen: print(space, 'SILENCE tag')
        return obj
        #return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o, handle, space + ' ')) for n, o in obj.items())
    return list(tokenize(o, handle, space + ' ') for o in obj)

with open(personachat_tab_file,'w') as z:

    dataset2 = tokenize(dataset, z)

print('---')
