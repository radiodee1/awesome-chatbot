#!/usr/bin/python3

import json

personachat_file = './raw/personachat_self_original.json'

personachat_tab_file = './data/raw.txt'

with open(personachat_file, "r", encoding="utf-8") as f:
    dataset = json.loads(f.read())


# Tokenize and encode the dataset using our loaded GPT tokenizer
def tokenize(obj, handle):

    if isinstance(obj, str):
        print(obj)
        if not (obj.startswith('__') or obj.endswith('__')):
            handle.write(obj + '\n')
        else:
            print('SILENCE tag')
        return obj
        #return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o, handle)) for n, o in obj.items())
    return list(tokenize(o, handle) for o in obj)

with open(personachat_tab_file,'w') as z:

    dataset2 = tokenize(dataset, z)

print('---')
