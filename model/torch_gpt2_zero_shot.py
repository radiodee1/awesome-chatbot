#!/usr/bin/python3.6

#@article{radford2019language,
#  title={Language Models are Unsupervised Multitask Learners},
#  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
#  year={2019}
#}

import sys
sys.path.append('..')
from model.settings import hparams
import torch
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import random

import logging
logging.basicConfig(level=logging.INFO)

gpt2_dir = 'gpt2' #hparams['data_dir'] + '/' + 'gpt2' + '/'

## vocab.json merges.txt

class Lang:
    def __init__(self, name, limit=None):
        self.name = name
        self.limit = limit
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if self.limit is None or self.n_words < self.limit :
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1


class NMT:
    def __init__(self):

        self.tokenizer = None
        self.model = None
        self.wordcount = 1
        self.words_end = ['.', '?', '!', '"']

        self.output_lang = None

        self.past = None

    def setup_for_interactive(self):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_dir)

        # Load pre-trained model (weights)
        self.model = GPT2LMHeadModel.from_pretrained(gpt2_dir)
        self.model.eval()

        print(self.tokenizer.__len__(),'max')
        self.output_lang = Lang('lang')
        for i in range(self.tokenizer.__len__()):
            self.output_lang.addWord(self.tokenizer.decode([i]))

    def get_sentence(self, i):
        num = 0
        text_1 = i
        text_2 = ""
        self.past = None
        #decode_list = []

        if False:
            word = self.random_word()
        else:
            word = ''
        space_character = ' ' ## no space!!??

        while num < self.wordcount:

            indexed_tokens_2 = self.tokenizer.encode(word + space_character + text_1 + ' ? ')

            tokens_tensor_2 = torch.tensor([indexed_tokens_2])

            with torch.no_grad():
                predictions_1, self.past = self.model(tokens_tensor_2, past=self.past)

            zlist = ''
            xlist = ''
            for i in range(predictions_1.size(1)):
                ii = i #0 ## i
                p_index = torch.argmax(predictions_1[0, ii, :], dim=-1).item()
                p_token = self.tokenizer.decode([p_index])
                zlist += '[' + str(p_index) + ']'
                xlist += p_token
            print()
            xlist = xlist.strip()
            xlist = xlist.replace(',','')
            xlist = xlist.replace('.','')
            print(zlist)

            print('out >',xlist)

            num += 1

            return xlist

    def loop(self):

        self.past = None

        while True:
            #num = 0
            try:
                text_1 = input(">>> ")
            except EOFError:
                print('\nend of file.')
                exit()
            print(text_1)
            self.get_sentence(text_1)


    def random_word(self):
        #word = self.tokenizer.decode([random.randint(0,self.tokenizer.__len__() -1)])
        word = self.output_lang.index2word[random.randint(0, self.output_lang.n_words - 1)]
        word = word.encode( "utf-8")
        word = word.decode("utf-8").strip()
        #print(word)
        return word

if __name__ == '__main__':
    g = NMT()
    g.setup_for_interactive()
    #print(g.tokenizer.encode(' '))
    #print(g.tokenizer.decode(g.tokenizer.encode(' ')))
    g.loop()
