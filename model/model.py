#!/usr/bin/python3

import numpy as np
from settings import hparams
from keras.preprocessing import text, sequence
from keras.models import Sequential , Model
from keras.layers import Embedding, Input, LSTM, Bidirectional, TimeDistributed, Flatten, dot
from keras.layers import Activation, RepeatVector, Permute, Merge, Dense #, TimeDistributedMerge
from keras.layers import Concatenate, Add, Multiply
#from keras.engine.topology import merge
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


if True:
    print ("stage: load model")
    word2vec_book = w2v.Word2Vec.load(os.path.join("../data", raw_embedding_filename + "_1.w2v"))
    words = len(word2vec_book.wv.vocab)
    vocab_size = words

if True:
    zzz_fr = None
    zzz_to = None
    with open(vocab_fr,'r') as r:
        zzz_fr = r.read()
        text_zzz_fr = []
        for x in zzz_fr.split('\n'):
            text_zzz_fr.append(x)

    with open(vocab_to, 'r') as r:
        zzz_to = r.read()
        text_zzz_to = []
        for x in zzz_to.split('\n'):
            text_zzz_to.append(x)

    with open(text_fr, 'r') as r:
        xxx = r.read()
        text_xxx = []
        for xx in xxx.split('\n'):
            text_xxx.append(xx)

    with open(text_to, 'r') as r:
        yyy = r.read()
        text_yyy = []
        for xx in yyy.split('\n'):
            text_yyy.append(xx)


if True:
    tokenize_voc_fr = text.Tokenizer(num_words=words,oov_token=oov_token, filters='\n' )
    tokenize_voc_fr.fit_on_texts(text_zzz_fr)

    tokenize_voc_to = text.Tokenizer(num_words=words, oov_token=oov_token, filters='\n')
    tokenize_voc_to.fit_on_texts(text_zzz_to)

    tokenize_text_fr = text.Tokenizer(num_words=words, oov_token=oov_token, filters='\n')
    tokenize_text_fr.fit_on_texts(text_zzz_to)

    tokenize_text_to = text.Tokenizer(num_words=words, oov_token=oov_token, filters='\n')
    tokenize_text_to.fit_on_texts(text_zzz_to)


def word_and_vector_size_arrays(text_xxx, text_yyy, double_y=False):
    ls_xxx = np.array([])
    ls_yyy = np.array([])

    temp_xxx = np.array([])
    temp_yyy = np.array([])

    mult = 1
    if double_y:
        mult = 2

    for ii in range(len(text_xxx)):
        ############### x #######################
        i = text.text_to_word_sequence(text_xxx[ii])
        ls = np.array([])
        for word in range(tokens_per_sentence):  # i:
            if word + 1 <= len(i):
                if i[word] in word2vec_book.wv.vocab:  # tokenize_voc_fr.word_index:
                    w = np.array([word2vec_book.wv.vocab[i[word]].index])
                    if ls.shape[0] == 0:
                        ls = w
                    else:
                        ls = np.hstack((ls, w))

                else:
                    pad = np.zeros(1)
                    if ls.shape[0] == 0:
                        ls = pad
                    else:
                        ls = np.hstack((ls, pad))
            else:
                pad = np.zeros(1)
                if ls.shape[0] == 0:
                    ls = pad
                else:
                    ls = np.hstack((ls, pad))
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
                    w = word2vec_book.wv[j[word]]
                    if double_y:
                        reverse = w[::-1]
                        w = np.hstack((w, reverse))
                    if ls.shape[0] == 0:
                        ls = w
                    else:
                        ls = np.vstack((ls, w))
                else:
                    pad = np.zeros(units * mult)
                    if ls.shape[0] == 0:
                        ls = pad
                    else:
                        ls = np.vstack((ls, pad))
            else:
                pad = np.zeros(units * mult)
                if ls.shape[0] == 0:
                    ls = pad
                else:
                    ls = np.vstack((ls, pad))
            pass

        if ls_yyy.shape[0] == 0:
            ls_yyy = ls
        else:
            ls_yyy = np.dstack((ls_yyy, ls))
        ############ batch #############

        if ii % (len(text_xxx) // batch_size) == 0:
            if temp_xxx.shape[0] == 0:
                temp_xxx = ls_xxx
            else:
                temp_xxx = np.dstack((temp_xxx, ls_xxx))
            ls_xxx = np.array([])
            ###############
            if temp_yyy.shape[0] == 0:
                temp_yyy = ls_yyy
            else:
                temp_yyy = np.dstack((temp_yyy, ls_yyy))
            ls_yyy = np.array([])

    return temp_xxx, temp_yyy






def get_batch(batch, x, y):
    """ first batch starts at 0 """
    a = batch_size * batch
    b = batch_size * (batch + 1)
    x = x[:,:,a:b]
    y = y[:,:,a:b]
    return x, y


def swap_axes(x, y):
    x = np.swapaxes(x, 0, 2)
    y = np.swapaxes(y, 0, 2)
    y = np.swapaxes(y, 1, 2)

    x = x[:,:,0]
    return x, y


def embedding_model_lstm():
    embedding_matrix = np.zeros((len(word2vec_book.wv.vocab), units))
    for i in range(len(word2vec_book.wv.vocab)):
        embedding_vector = word2vec_book.wv[word2vec_book.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    x_shape = (batch_size, tokens_per_sentence)

    valid_word = Input(shape=x_shape[1:])

    embeddings_a = Embedding(words, units, weights=[embedding_matrix],
                           batch_size=batch_size, input_shape=x_shape[1:])
    embed_a = embeddings_a(valid_word)

    lstm_a = Bidirectional(LSTM(units=units,
                                input_shape=(tokens_per_sentence, units),
                                return_sequences=True))

    recurrent_a = lstm_a(embed_a)

    lstm_a2 = Bidirectional(LSTM(units=units,
                                input_shape=(tokens_per_sentence,units),
                                return_sequences=True))

    recurrent_a2 = lstm_a2(recurrent_a)

    time_dist_a = TimeDistributed(Dense(units, activation='softmax'))(recurrent_a2)


    '''
    attention_b = Input(shape=x_shape[1:])
    dense_b = Dense(input_dim=x_shape[1:], units=tokens_per_sentence)
    dense_out = dense_b(attention_b)
    #dense_out = dense_b(recurrent_a)
    activation_b = Activation('softmax')
    act_out = activation_b(dense_out)
    repeat_vec_b = RepeatVector(units)
    repeat_b = repeat_vec_b(act_out)
    permute_b = Permute((2,1))
    end_b = permute_b(repeat_b)

    merge_c = Add()([recurrent_a,end_b])

    time_distributed_c = TimeDistributed(merge_c)
    #time_c = time_distributed_c(merge_c)
    '''

    k_model = Model(inputs=[valid_word], outputs=[recurrent_a2])

    k_model.compile(optimizer='adam', loss='mse')

    return k_model

def lstm_model_api():
    input_dim = tokens_per_sentence
    hidden = units

    # The LSTM  model -  output_shape = (batch, step, hidden)
    model1 = Sequential()
    model1.add(LSTM(units=units, output_dim=hidden, input_shape=(tokens_per_sentence, units), return_sequences=True))

    # The weight model  - actual output shape  = (batch, step)
    # after reshape : output_shape = (batch, step,  hidden)
    model2 = Sequential()
    model2.add(Dense(input_dim=input_dim, output_dim=step))
    model2.add(Activation('softmax'))  # Learn a probability distribution over each  step.
    # Reshape to match LSTM's output shape, so that we can do element-wise multiplication.
    model2.add(RepeatVector(hidden))
    model2.add(Permute(2, 1))

    # The final model which gives the weighted sum:
    model = Sequential()
    model.add(Merge([model1, model2], 'mul'))  # Multiply each element with corresponding weight a[i][j][k] * b[i][j]
    model.add(TimeDistributedMerge('sum'))  # Sum the weighted elements.

    model.compile(loss='mse', optimizer='sgd')

    return model

def train_embedding_model_api(model, x, y):
    z = x.shape[2] // batch_size
    for i in range(z):
        xx , yy = get_batch(i, x, y)
        xx, yy = swap_axes(xx, yy)
        #print (xx.shape)
        model.train_on_batch(xx,yy)


x, y = word_and_vector_size_arrays(text_xxx, text_yyy, double_y=True)


model = embedding_model_lstm()

train_embedding_model_api(model, x, y)

print ("here")