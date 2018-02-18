#!/usr/bin/python3

import numpy as np
from settings import hparams
from keras.preprocessing import text
from keras.models import Sequential
from keras.layers import Embedding, Input, LSTM, Bidirectional, TimeDistributed, Flatten

print(hparams)

words = hparams['num_vocab_total']
text_fr = hparams['data_dir'] + hparams['test_name'] + '.' + hparams['src_ending']
text_to = hparams['data_dir'] + hparams['test_name'] + '.' + hparams['tgt_ending']

vocab_fr = hparams['data_dir'] + hparams['vocab_name'] + '.' + hparams['src_ending']
vocab_to = hparams['data_dir'] + hparams['vocab_name'] + '.' + hparams['tgt_ending']
oov_token = hparams['unk']
batch_size = hparams['batch_size']
units = hparams['units']

zzz_fr = None
zzz_to = None
with open(vocab_fr,'r') as r:
    zzz_fr = r.read()
    text_zzz_fr = []
    for x in zzz_fr.split('\n'):
        text_zzz_fr.append(x)

#print (text_zzz_fr)

with open(vocab_to, 'r') as r:
    zzz_to = r.read()
    text_zzz_to = []
    for x in zzz_to.split('\n'):
        text_zzz_to.append(x)

with open(text_fr, 'r') as r:
    xxx = r.read()
    text_xxx = []
    for x in xxx.split('\n'):
        text_xxx.append(x)

with open(text_to, 'r') as r:
    yyy = r.read()
    text_yyy = []
    for x in yyy.split('\n'):
        text_yyy.append(x)





if True:
    tokenize_voc_fr = text.Tokenizer(num_words=words,oov_token=oov_token, filters='\n' )#, filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n')
    tokenize_voc_fr.fit_on_texts(text_zzz_fr)
    #x = tokenize_voc_fr.texts_to_matrix(text_zzz_fr)

    print (tokenize_voc_fr.word_index, len(tokenize_voc_fr.word_index))
    exit()


    tokenize_voc_to = text.Tokenizer(num_words=words, oov_token=oov_token, filters='\n')
    tokenize_voc_to.fit_on_texts(text_zzz_to)
    #y = tokenize_voc_to.texts_to_matrix(text_zzz_to)

if True:

    ls_xxx = [] #np.array([])
    ls_yyy = [] #np.array([])

    temp_xxx = [] #np.array([])
    temp_yyy = [] #np.array([])

    for ii in range(len(text_xxx)):
        i = text.text_to_word_sequence(text_xxx[ii])
        ls = [] #[hparams['sol']]# np.array([])
        for word in i:
            if word in tokenize_voc_fr.word_index:
                #word = np.array([tokenize_voc_fr.word_index[word]])
                ls.append(tokenize_voc_fr.word_index[word])
                #ls = np.hstack((ls, word))
            pass
        #ls = np.asarray(ls)
        ls_xxx.extend(ls)
        #ls_xxx.append(hparams['eol'])
        #ls_xxx = np.stack((ls_xxx,ls))
        j = text.text_to_word_sequence(text_yyy[ii])
        ls = [] #[hparams['sol']] #np.array([])
        for word in j:
            if word in tokenize_voc_to.word_index:
                #word = np.array([tokenize_voc_to.word_index[word]])
                ls.append(tokenize_voc_to.word_index[word])
                #ls = np.hstack((ls, word))
            pass
        #ls = np.asarray(ls)
        ls_yyy.extend(ls)
        #ls_yyy.append(hparams['eol'])
        #ls_yyy = np.stack((ls_yyy,ls))

        if ii % batch_size  == 0:
            temp_xxx.extend(ls_xxx)
            #temp_xxx = np.dstack((temp_xxx,ls_xxx))
            ls_xxx = []# np.array([])
            temp_yyy.extend(ls_yyy)
            #temp_yyy = np.dstack((temp_yyy,ls_yyy))
            ls_yyy = []# np.array([])

temp_yyy.extend([0,0,0,0,0])

x = np.array(temp_xxx)
y = np.array(temp_yyy)

#print (len(x),'\n', len(x[2]))
#print (len(x[0][0]))


if False:
    #x = np.array(x)
    #y = np.array(y)


    x = np.reshape(x, (batch_size,))
    y = np.reshape(y, (batch_size,))
    #x = np.transpose(x)
    #y = np.transpose(y)

#print (x)
#print (x.shape[1:])
#print (x.shape)
x_shape = (1,1)# x.shape
#x_shape =(x.shape[0], x.shape[0])

model = Sequential()
model.add(Embedding(words, units, input_shape=x_shape[1:]))
#model.add(Flatten())
shape = x_shape[1:][0]
model.add(LSTM(shape, return_sequences=True))
model.add(Bidirectional(LSTM(shape)))

#model.add(LSTM(units))

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x ,y,epochs=1, batch_size=batch_size)