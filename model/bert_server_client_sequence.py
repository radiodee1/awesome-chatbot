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

if False:
    with open(hparams['data_dir'] + '/bert_data/uncased_L-12_H-768_A-12/vocab.txt') as fp:
        for line in fp.readlines():
            q1 = line.split('\t')[0].strip()
            questions1.append(q1 )
            #questions2.append(q3)
        print(questions1[:1000])

doc_vecs = None
bc = BertClient()
if len(questions1) > 0:
    doc_vecs = bc.encode(questions1)
answer = []

while True:
    query = input('your question: ') #.split()
    query = [ query ]
    print(query)
    query_vec = bc.encode(query, show_tokens=True)#[0]
    print(len(query_vec[:]))
    print(query_vec)
    #exit()

    query_vec = query_vec[-1]
    # compute normalized dot product as score
    if doc_vecs is None: continue
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    for idx in topk_idx:
        print('> %s\t%s' % (score[idx], questions1[idx]  )) # (questions1[idx] + ' | ' + questions2[idx])))
    #answer = answer + [ questions1[ topk_idx[0] ] ]