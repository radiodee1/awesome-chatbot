#!/usr/bin/python3

import os
import sys
sys.path.append('..')
import requests
from tqdm import tqdm
from model.settings import hparams

#hparams['data_dir'] = '../data/'

if len(sys.argv) != 2:
    print('You must enter the model name as a parameter, e.g.: download_model.py 117M')
    sys.exit(1)

model = sys.argv[1]

if model == '117M':
    savedir = ''
else:
    savedir = '/' + model + '/'

subdir = os.path.join('models', model)
subdir2 = os.path.join(hparams['data_dir'], 'tf_gpt2_data') + "/" + savedir

if not os.path.exists(subdir2):
    print(subdir2)
    os.makedirs(subdir2)
subdir2 = subdir2.replace('\\','/') # needed for Windows


for filename in ['checkpoint','encoder.json','hparams.json','model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']:

    r = requests.get("https://storage.googleapis.com/gpt-2/" + subdir + "/" + filename, stream=True)

    with open(os.path.join(subdir2, filename), 'wb') as f:
        file_size = int(r.headers["content-length"])
        chunk_size = 1000
        with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
            # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size)
