#!/usr/bin/python3

import numpy as np
from settings import hparams
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Embedding, Input, LSTM, Bidirectional, TimeDistributed, Flatten
import gensim.models.word2vec as w2v
import os
import tensorflow as tf
#print(hparams)

words = hparams['num_vocab_total']
text_fr = hparams['data_dir'] + hparams['test_name'] + '.' + hparams['src_ending']
text_to = hparams['data_dir'] + hparams['test_name'] + '.' + hparams['tgt_ending']

vocab_fr = hparams['data_dir'] + hparams['vocab_name'] + '.' + hparams['src_ending']
vocab_to = hparams['data_dir'] + hparams['vocab_name'] + '.' + hparams['tgt_ending']
oov_token = hparams['unk']
batch_size = hparams['batch_size']
units = hparams['units']
tokens_per_sentence = hparams['tokens_per_sentence']
raw_embedding_filename = hparams['raw_embedding_filename']

tf.flags.DEFINE_integer("batch_size", batch_size, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 8, "Batch size during evaluation")

if True:
    print ("stage: load model")
    word2vec_book = w2v.Word2Vec.load(os.path.join("../data", raw_embedding_filename + "_1.w2v"))
    words = len(word2vec_book.wv.vocab)
    vocab_size = words
    print(word2vec_book.wv.index2word[vocab_size - 1], word2vec_book.wv.index2word[vocab_size - 2], word2vec_book.wv.index2word[vocab_size - 3])

if True:
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
        for xx in xxx.split('\n'):
            #for x in xx.split():
            text_xxx.append(xx)

    with open(text_to, 'r') as r:
        yyy = r.read()
        text_yyy = []
        for xx in yyy.split('\n'):
            #for x in xx.split():
            text_yyy.append(xx)


if True:
    tokenize_voc_fr = text.Tokenizer(num_words=words,oov_token=oov_token, filters='\n' )#, filters='"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n')
    tokenize_voc_fr.fit_on_texts(text_zzz_fr)
    #x_unused = tokenize_voc_fr.texts_to_matrix(text_zzz_fr)


    #print (tokenize_voc_fr.word_index, len(tokenize_voc_fr.word_index))


    tokenize_voc_to = text.Tokenizer(num_words=words, oov_token=oov_token, filters='\n')
    tokenize_voc_to.fit_on_texts(text_zzz_to)
    #y = tokenize_voc_to.texts_to_matrix(text_zzz_to)

    tokenize_text_fr = text.Tokenizer(num_words=words, oov_token=oov_token, filters='\n')
    tokenize_text_fr.fit_on_texts(text_zzz_to)
    #x_matrix = tokenize_text_fr.texts_to_matrix(text_xxx)
    #print (x_matrix, x_matrix.shape)
    #x_matrix = sequence.pad_sequences(text_xxx, maxlen=tokens_per_sentence)

    #print (x_matrix, x_matrix.shape)
    #x_matrix = np.expand_dims(x_matrix, axis=0)

    tokenize_text_to = text.Tokenizer(num_words=words, oov_token=oov_token, filters='\n')
    tokenize_text_to.fit_on_texts(text_zzz_to)
    #y_matrix = tokenize_text_to.texts_to_matrix(text_yyy)

    #y_matrix = np.dstack((y_matrix,y_matrix))

    #print (x_matrix.shape)

def word_and_vector_size_arrays(text_xxx, text_yyy):
    ls_xxx = np.array([])
    ls_yyy = np.array([])

    temp_xxx = np.array([])
    temp_yyy = np.array([])

    for ii in range(len(text_xxx)):
        ############### x #######################
        i = text.text_to_word_sequence(text_xxx[ii])
        ls = np.array([])
        for word in range(tokens_per_sentence):  # i:
            if word + 1 <= len(i):
                if i[word] in word2vec_book.wv.vocab:  # tokenize_voc_fr.word_index:
                    w = np.array([tokenize_voc_fr.word_index[i[word]]])
                    #w = word2vec_book.wv[i[word]]
                    if ls.shape[0] == 0:
                        ls = w
                    else:
                        ls = np.hstack((ls, w))

                else:
                    pad = np.zeros(1)
                    # ls = np.vstack((ls,pad))
                    if ls.shape[0] == 0:
                        ls = pad
                    else:
                        ls = np.hstack((ls, pad))
                    # ls.append(0)
            else:
                # ls.append(0)
                pad = np.zeros(1)
                if ls.shape[0] == 0:
                    ls = pad
                else:
                    ls = np.hstack((ls, pad))
                # ls = np.vstack((ls, pad))
            pass

        if ls_xxx.shape[0] == 0:
            ls_xxx = ls

        else:
            ls_xxx = np.dstack((ls_xxx, ls))

        ################# y ####################
        j = text.text_to_word_sequence(text_yyy[ii])
        ls = np.array([])
        for word in range(tokens_per_sentence):  # j:
            if word + 1 <= len(j):
                if j[word] in word2vec_book.wv.vocab:  # tokenize_voc_to.word_index:
                    # w = np.array([tokenize_voc_to.word_index[j[word]]])
                    w = word2vec_book.wv[j[word]]
                    # ls.append(tokenize_voc_to.word_index[j[word]])
                    if ls.shape[0] == 0:
                        ls = w
                    else:
                        ls = np.vstack((ls, w))
                    # ls = np.vstack((ls, w))
                else:
                    # ls.append(0)
                    pad = np.zeros(units)
                    if ls.shape[0] == 0:
                        ls = pad
                    else:
                        ls = np.vstack((ls, pad))
                    # ls = np.vstack((ls, pad))
            else:
                # ls.append(0)
                pad = np.zeros(units)
                if ls.shape[0] == 0:
                    ls = pad
                else:
                    ls = np.vstack((ls, pad))
                # ls = np.vstack((ls, pad))
            pass
        # ls = np.asarray(ls)
        # ls_yyy.extend(ls)
        # ls_yyy.append(hparams['eol'])
        if ls_yyy.shape[0] == 0:
            ls_yyy = ls
        else:
            ls_yyy = np.dstack((ls_yyy, ls))
        ls = np.array([])
        ############ batch #############

        if ii % batch_size == 0:
            '''
            # exit()
            if temp_xxx.shape[0] == 0:
                temp_xxx = ls_xxx
            else:
                temp_xxx = np.vstack((temp_xxx, ls_xxx))
            ls_xxx = np.array([])
            '''
            # temp_yyy.extend(ls_yyy)
            if temp_yyy.shape[0] == 0:
                temp_yyy = ls_yyy
            else:
                temp_yyy = np.dstack((temp_yyy, ls_yyy))
            ls_yyy = np.array([])

            ls = np.array([])
    return ls_xxx, temp_yyy


def vector_size_arrays(text_xxx, text_yyy):

    ls_xxx = np.array([])
    ls_yyy = np.array([])

    temp_xxx = np.array([])
    temp_yyy = np.array([])

    for ii in range(len(text_xxx)):
        ############### x #######################
        i = text.text_to_word_sequence(text_xxx[ii])
        ls = np.array([])
        for word in range(tokens_per_sentence):#i:
            if word + 1 <= len(i) :
                if i[word] in word2vec_book.wv.vocab: # tokenize_voc_fr.word_index:
                    #w = np.array([tokenize_voc_fr.word_index[i[word]]])
                    w = word2vec_book.wv[i[word]]
                    if ls.shape[0] == 0:
                        ls = w
                    else:
                        ls = np.vstack((ls, w))

                else:
                    pad = np.zeros(units)
                    #ls = np.vstack((ls,pad))
                    if ls.shape[0] == 0:
                        ls = pad
                    else:
                        ls = np.vstack((ls, pad))
                    #ls.append(0)
            else:
                #ls.append(0)
                pad = np.zeros(units)
                if ls.shape[0] == 0:
                    ls = pad
                else:
                    ls = np.vstack((ls, pad))
                #ls = np.vstack((ls, pad))
            pass
        #ls = np.asarray(ls)
        #ls_xxx.extend(ls)
        #ls_xxx.append(hparams['eol'])
        if ls_xxx.shape[0] == 0:
            ls_xxx = ls

        else:
            ls_xxx = np.dstack((ls_xxx,ls))

        ################# y ####################
        j = text.text_to_word_sequence(text_yyy[ii])
        ls = np.array([])
        for word in range(tokens_per_sentence):#j:
            if word + 1 <= len(j):
                if j[word] in word2vec_book.wv.vocab: # tokenize_voc_to.word_index:
                    #w = np.array([tokenize_voc_to.word_index[j[word]]])
                    w = word2vec_book.wv[j[word]]
                    #ls.append(tokenize_voc_to.word_index[j[word]])
                    if ls.shape[0] == 0:
                        ls = w
                    else:
                        ls = np.vstack((ls, w))
                    #ls = np.vstack((ls, w))
                else:
                    #ls.append(0)
                    pad = np.zeros(units)
                    if ls.shape[0] == 0:
                        ls = pad
                    else:
                        ls = np.vstack((ls, pad))
                    #ls = np.vstack((ls, pad))
            else:
                #ls.append(0)
                pad = np.zeros(units)
                if ls.shape[0] == 0:
                    ls = pad
                else:
                    ls = np.vstack((ls, pad))
                #ls = np.vstack((ls, pad))
            pass
        #ls = np.asarray(ls)
        #ls_yyy.extend(ls)
        #ls_yyy.append(hparams['eol'])
        if ls_yyy.shape[0] == 0:
            ls_yyy = ls
        else:
            ls_yyy = np.dstack((ls_yyy,ls))
        ls = np.array([])
        ############ batch #############

        if ii % batch_size  == 0:

            #exit()
            if temp_xxx.shape[0] == 0:
                temp_xxx = ls_xxx
            else:
                temp_xxx = np.dstack((temp_xxx,ls_xxx))
            ls_xxx = np.array([])
            #temp_yyy.extend(ls_yyy)
            if temp_yyy.shape[0] == 0:
                temp_yyy = ls_yyy
            else:
                temp_yyy = np.dstack((temp_yyy,ls_yyy))
            ls_yyy = np.array([])

            ls = np.array([])

    return temp_xxx, temp_yyy

x, y = word_and_vector_size_arrays(text_xxx, text_yyy)


if False:
    x = np.array(temp_xxx)
    y = np.array(temp_yyy)



if False:
    x = np.swapaxes(x, 0, 2)
    y = np.swapaxes(y, 0, 2)

    #x = np.swapaxes(x, 0,1)
    y = np.swapaxes(y, 1,2)

    #x = np.expand_dims(x, axis=0)
    #y = np.expand_dims(y, axis=0)

print (x.shape , y.shape)

if False:
    a = 0
    x = np.array_split(x, 16, axis=a)
    y = np.array_split(y, 16, axis=a)

    #print (x.shape , y.shape)



x_shape =  x[0].shape #_matrix.shape

#x_shape = (batch_size, batch_size, tokens_per_sentence)
print (x_shape)


if True:
    embedding_matrix = np.zeros((len(word2vec_book.wv.vocab), units))
    for i in range(len(word2vec_book.wv.vocab)):
        embedding_vector = word2vec_book.wv[word2vec_book.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(words, units,
                    #input_dim=units,
                    #output_dim=units,
                    weights=[embedding_matrix],
                    input_length=tokens_per_sentence,
                    batch_size=batch_size ,
                    input_shape=x_shape[1:],
                    #batch_input_shape=x_shape[1:]
                    ))

#shape = x_shape[1:][0]
#model.add(LSTM(batch_size, input_shape=x_shape[1:], batch_size=batch_size, return_sequences=True))
#model.add(Bidirectional(LSTM(shape)))

#model.add(LSTM(units))

model.compile(optimizer='rmsprop', loss='mse')

#model.fit(x[0] ,y[0],epochs=1, batch_size=batch_size)

print (x[0].shape)

model.train_on_batch(x, y)