#!/usr/bin/python3.6

import numpy as np
from bert_serving.client import BertClient
import sys
sys.path.append("..")
from model.settings import hparams

topk = 5

bc = BertClient()
questions = []

with open(hparams['data_dir'] + '/t2t_data/raw.txt') as fp:
    for line in fp.readlines():
        q1 = line.split('\t')[0].strip()
        q2 = line.split('\t')[1].strip()
        if len(q1) > 0: questions.append(q1)
        if len(q2) > 0: questions.append(q2)
        #print(questions)

doc_vecs = bc.encode(questions[:1000])

while True:
    query = input('your question: ')
    query_vec = bc.encode([query])#[0]
    #print(query_vec)
    query_vec = query_vec[0]
    # compute normalized dot product as score
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    for idx in topk_idx:
        print('> %s\t%s' % (score[idx], questions[idx]))