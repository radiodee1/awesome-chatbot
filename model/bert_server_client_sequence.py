#!/usr/bin/python3.6

import numpy as np
from bert_serving.client import BertClient
import sys
sys.path.append("..")
from model.settings import hparams

topk = 5
#word_level = False

#bc = BertClient()
questions1 = []
questions2 = []

with open(hparams['data_dir'] + '/t2t_data/raw.txt') as fp:
    for line in fp.readlines():
        qq = []
        q1 = line.split('\t')[0].strip()
        q2 = line.split('\t')[1].strip()
        q3 = ""
        for i in q2.split(' '):
            q3 += i + ' '
            if len(q1) > 0 and len(q3.split(' ')) > 2:
                #qq = [ q1 , q2]
                questions1.append(q1 + " " + q3)
                questions2.append(q3)
    print(questions1[:1000])

bc = BertClient()
doc_vecs = bc.encode(questions1[:1000])

while True:
    query = input('your question: ')
    query_vec = bc.encode([query])#[0]

    query_vec = query_vec[0]
    # compute normalized dot product as score
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    for idx in topk_idx:
        print('> %s\t%s' % (score[idx], (questions1[idx] + ' | ' + questions2[idx])))