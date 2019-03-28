#!/usr/bin/python3.6

import numpy as np
from bert_serving.client import BertClient

topk = 5

bc = BertClient()

questions = ['First do it', 'then do it right', 'then do it better']
doc_vecs = bc.encode(questions)

while True:
    query = input('your question: ')
    query_vec = bc.encode([query])[0]
    # compute normalized dot product as score
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    for idx in topk_idx:
        print('> %s\t%s' % (score[idx], questions[idx]))