#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
import codecs

import glob

import logging

import multiprocessing

import os

import pprint
import io
import re

#import nltk
#from nltk.tokenize import TweetTokenizer, sent_tokenize, PunktSentenceTokenizer
#from nltk.stem import *
import gensim.models.word2vec as w2v
from settings import hparams



#########################################

def sentence_to_wordlist(raw, sentence_label="", pos_tag=False):
    pre = raw #raw.split()
    raw = ""
    #w = []
    if not (type(pre) is list): return [pre.lower()]
    for x in pre:
        if not x.endswith( u"'s"):
            #w.append(x)
            raw = raw + " " + x
        else:
            #print("missed s")
            pass
    clean = re.sub("[^a-zA-Z]"," ", raw)

    words = clean.split()
    words = [x.lower() for x in words]

    if pos_tag:
        tag = []
        words = nltk.pos_tag(words)
        for word in words:
            tag.append(word[0])
            tag.append(word[1].decode("utf-8"))
        words = tag
        #print (words)
        pass

    if len(sentence_label) > 0: words.append(sentence_label)
    #print (words)
    return words

########################################

test = []

if True:
    test = [["I go to school."],[" I've gone to school."],[" go north."],[" go south."],[" go east."],[" go west."],[" move south."]]


    new_test = []
    for t in test:
        z = sentence_to_wordlist(t, pos_tag=False)
        new_test.append(z)
        #print (z)
    test = new_test
    print (test)

#exit()

###########################################

def assemble_corpus(glob_txt, stem_words=False, sentence_label="", pos_tag=False, tweet_tag=False, print_sentences=False):
    pass

    #add everything once

    #add zork text twice more
    book_filenames = sorted(glob.glob(glob_txt))

    #book_filenames.extend(sorted(glob.glob("data/z*.txt")))

    print (book_filenames)

    print ("stage: start")

    corpus_raw = [u""]
    for book_filename in book_filenames:
        print("stage: Reading '{0}'...".format(book_filename))
        with io.open(book_filename, "r", encoding="utf-8") as book_file: ## codecs.open
            for line in book_file:
                #corpus_raw += line
                corpus_raw.append(line)
            #corpus_raw += book_file.read()
        print("stage: Corpus is now {0} characters long".format(len(corpus_raw)))
        print()

    if isinstance(corpus_raw, list):
        pre_sent = corpus_raw
    else:
        pre_sent = sent_tokenize(corpus_raw)

    corpus_raw = u""

    #########################

    if tweet_tag:
        tokenizer = TweetTokenizer()

        print ("stage: tweet")

        post_sent = []
        for i in pre_sent:
            raw_sentences = tokenizer.tokenize(i) ##tweet style

            post_sent.append(raw_sentences)
            #print (raw_sentences)

        raw_sentences = post_sent
        post_sent = []

    else:
        raw_sentences = pre_sent
        pre_sent = []

    ###########################

    if pos_tag: print ("stage: pos tagging")
    else:
        pass
        #if print_sentences: print (raw_sentences[-10:])
        #return raw_sentences

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            if not type(raw_sentence) == list: raw_sentence = raw_sentence.split()
            z = sentence_to_wordlist(raw_sentence, sentence_label=sentence_label, pos_tag=pos_tag)
            if len(z) > 0:
                sentences.append(z)

    # stem words and add them
    if stem_words:
        print ("stage: stem")
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sent = sentence_to_wordlist(raw_sentence)
                sent = [stemmer.stem(word) for word in sent]
                sentences.append(sent)

    if print_sentences: print(sentences[-20:])
    #print(sentence_to_wordlist(raw_sentences[0], pos_tag=pos_tag))

    token_count = sum([len(sentence) for sentence in sentences])
    print("stage: The corpus contains {0:,} tokens".format(token_count))

    return sentences

####################################################
model_generate_new = True

game_glob1 = "../raw/zork1-output.txt" ## actual commands processed
game_glob2 = "../raw/z*.txt" ## not for good game corpus
game_glob3 = "../raw/wiki*.txt"
game_glob4 = "../raw/zork1-example.txt"

sentences_book = []
if False:
    sentences_game = assemble_corpus(game_glob1, stem_words=False)

if True:
    sentences_zork = assemble_corpus(game_glob2, pos_tag=False,print_sentences=True)

if False:
    #sentences_book = []
    sentences_book = assemble_corpus(game_glob3, pos_tag=False)

if False:
    #sentences_book = []
    sentences_book = assemble_corpus(game_glob4, pos_tag=False)

if True:
    sentences_book.extend(sentences_zork)
    sentences_book.extend(test)

#print (sentences_book)

if False:
    #print ("-----------------")

    for sentences in sentences_book:
        xx = sentences
        yy = 0
        for i in range(len(xx)):
            yy += 1
            print (xx[i] , end="")
            if yy % 1000 == 0:
                print (". ")
                yy = 0
            elif not i == len(xx) - 1:
                print (" ", end="")
        print (". ")


    exit()


#exit()

raw_embedding_filename = hparams['raw_embedding_filename']
units = hparams['units']

############################################
num_features =  units #  900 is not good
# Minimum word count threshold.
min_word_count = 1 # 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7 # 7

# Downsample setting for frequent words.
#0 - 1e-5 is good for this
downsampling = 1e-2

# Seed for the RNG, to make the results reproducible.
#random number generator
#deterministic, good for debugging
seed = 1

epochs = 1000

if model_generate_new and True:
    word2vec_book = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )



elif False:
    print ("stage: load model")
    word2vec_book = w2v.Word2Vec.load(os.path.join("../data", raw_embedding_filename +"_1.w2v"))
    #word2vec_book = w2v.KeyedVectors.load_word2vec_format(os.path.join("trained",'saved_google',"GoogleNews-vectors-negative300.bin"),binary=True)


if True:
    word2vec_book.build_vocab(sentences_book)

    print("stage: Word2Vec vocabulary length:", len(word2vec_book.wv.vocab))

if True:
    print ("stage: train")

    word2vec_book.train(sentences_book,
                        total_examples=len(word2vec_book.wv.vocab),
                        epochs=epochs)



    word2vec_book.save(os.path.join("../data", raw_embedding_filename + "_1.w2v"))


