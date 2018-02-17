#!/usr/bin/python3

from settings import hparams
from keras.preprocessing import text


print(hparams)

words = hparams['num_vocab_total']
text_fr = hparams['data_dir'] + hparams['test_name'] + '.' + hparams['src_ending']
print(text_fr)
zzz = None
with open(text_fr,'r') as r:
    zzz = r.read()

tokenize_fr = text.Tokenizer(num_words=words,oov_token='<unk>', filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n')
tokenize_fr.fit_on_texts([zzz])
print(tokenize_fr.word_index)

z_in = []
for z in zzz.split(): # '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    #print(z)
    z_in.append(tokenize_fr.word_index[z])
#print(z_in)
