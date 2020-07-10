#!/usr/bin/env python3

from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

#prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
#next_sentence = "The sky is blue due to the shorter wavelength of blue light."

prompt = "that's great."
next_sentence = "that is great"

encoding = tokenizer(prompt, next_sentence, return_tensors='pt')
loss, logits = model(**encoding, next_sentence_label=torch.LongTensor([1]))

print(logits)
#assert logits[0, 0] < logits[0, 1] # next sentence was random