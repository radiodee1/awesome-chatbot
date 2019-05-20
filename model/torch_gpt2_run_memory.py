#!/usr/bin/python3

'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT

MIT License

Copyright (c) 2019 OpenAI, HugginFace Inc. team. and TaeHwan Jung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

import os
import sys
sys.path.append('torch_gpt2')
import torch
import random
import argparse
import numpy as np
import json
import re
import datetime
#from functools import lru_cache
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
#from GPT2.encoder import get_encoder
from GPT2.encoder import Encoder

realpath = os.path.dirname(os.path.realpath(__file__))

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
        self.args = None
        self.state_dict = None
        self.config = None
        self.device = None
        self.model = None
        self.enc = None

        self.output_lang = None

        self.common = ''
        self.previous_sentences = []
        self.gather_sentences = True
        self.recent_in = ''

    def setup_for_interactive(self):
        self.get_args()
        self.load_state_dict()
        self.load_model()

        ## this is not used but is required for bot software...
        self.output_lang = Lang('lang')
        for i in range(len(self.enc.encoder.items())):
            self.output_lang.addWord(self.enc.decode([i]))

        ## do this also with each input...
        self.prepare_common()

    def prepare_common(self):
        now = datetime.datetime.now()
        time = now.strftime("%H:%M")
        date = now.strftime("%B %d, %Y")
        self.common = 'My name is David. The time is ' + time + ' ' + date + '.'

    def get_sentence(self, i):
        if self.gather_sentences:
            s = []
            for k in self.previous_sentences:
                k = k.strip().strip('..')
                if not k.endswith('?'): k = k + '.'
                s.append(k)
            i = '\n\nQ: ' + i
            s.append(i)
            self.prepare_common()
            i = self.common + "\n\n" + ' '.join(s)
            print('',"+" * 10, '\n', i, '\n','+' * 10)
        i = self.prepare_input(i)

        self.args.text = i
        text = self.text_generator()
        text = self.prepare_output(text)
        print(text,"<")
        return text

    def loop(self):
        while True:
            try:
                i = input("> ")
                self.recent_in = i
                self.get_sentence(i)
            except EOFError:
                print()
                exit()
            except KeyboardInterrupt:
                print()
                exit()

    def prepare_input(self, i):
        if not self.gather_sentences:
            i = 'q: ' + i + '?'
        else:
            i = i + "?"
        return i

    def prepare_output(self, i):
        char_end = ['?','!']
        contains_junk = False
        char_junk = [i for i in '{[]}@$%^&#']
        out = []
        for ii in i:
            if ii.strip() != "" or ii == ' ':
                if ii not in ['*']:
                    out.append(ii)
            elif len(out) > 1:
                break
            if ii in char_end:
                break
            if ii in char_junk:
                contains_junk = True
                break
        i = ''.join(out)

        i = i.strip()
        if i.lower().startswith('a:'): i = i[len('a:'):]
        if i.lower().startswith('a :'): i = i[len('a :'):]

        if True:
            out = []
            for ii in i.split(' '):
                if ii not in out or ii.lower() in ['the', 'that'] or True: ## end with punctuation, not repeat!
                    out.append(ii)
                else:
                    break

                if (ii.endswith('.') or ii.endswith('!') or ii.endswith('?')) and len(ii) > 1 and ii.count('.') >= 1:
                    break
        i = ' '.join(out)

        i = re.sub('[/;\"]','',i)
        if contains_junk is True:
            i = ''

        if self.gather_sentences:
            i = i.strip()
            if i.lower().startswith('q:'): i = i[len('q:'):]
            if i.lower().startswith('q :'): i = i[len('q :'):]
            i = re.sub('[?!]', ' ', i)
            if i not in self.previous_sentences:
                self.previous_sentences.append(self.recent_in + "?")
                self.previous_sentences.append(i + ".")
        return i

    #########################################

    def get_encoder(self):
        with open(realpath +'/./torch_gpt2/GPT2/encoder.json', 'r') as f:
            encoder = json.load(f)
        with open(realpath + '/./torch_gpt2/GPT2/vocab.bpe', 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        return Encoder(
            encoder=encoder,
            bpe_merges=bpe_merges,
        )

    def get_args(self ):
        parser = argparse.ArgumentParser()
        parser.add_argument("--text", type=str, required=False)
        parser.add_argument("--quiet", type=bool, default=False)
        parser.add_argument("--nsamples", type=int, default=1)
        parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
        parser.add_argument("--batch_size", type=int, default=-1)
        parser.add_argument("--length", type=int, default=25)
        parser.add_argument("--temperature", type=float, default=0.01)
        parser.add_argument("--top_k", type=int, default=40)
        self.args = parser.parse_args()

    def load_model(self):
        if self.args.quiet is False:
            print(self.args)

        if self.args.batch_size == -1:
            self.args.batch_size = 1
        assert self.args.nsamples % self.args.batch_size == 0

        seed = random.randint(0, 2147483647)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Model
        self.enc = self.get_encoder()
        self.config = GPT2Config()
        self.model = GPT2LMHeadModel(self.config)
        self.model = load_weight(self.model, self.state_dict)
        self.model.to(self.device)
        self.model.eval()

    def text_generator(self):

        if self.args.length == -1:
            self.args.length = self.config.n_ctx // 2
        elif self.args.length > self.config.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % self.config.n_ctx)

        print(self.args.text)
        context_tokens = self.enc.encode(self.args.text)

        generated = 0
        for _ in range(self.args.nsamples // self.args.batch_size):
            out = sample_sequence(
                model=self.model, length=self.args.length,
                context=context_tokens  if not  self.args.unconditional else None,
                start_token=self.enc.encoder['<|endoftext|>'] if self.args.unconditional else None,
                batch_size=self.args.batch_size,
                temperature=self.args.temperature, top_k=self.args.top_k, device=self.device
            )
            out = out[:, len(context_tokens):].tolist()
            for i in range(self.args.batch_size):
                generated += 1
                text = self.enc.decode(out[i])
                if self.args.quiet is False:
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
        return text


    def load_state_dict(self):
        p = realpath + '/./torch_gpt2/gpt2-pytorch_model.bin'
        if os.path.exists(p):
            self.state_dict = torch.load(p, map_location='cpu' if not torch.cuda.is_available() else None)
            #self.text_generator(state_dict)
        else:
            print('Please download gpt2-pytorch_model.bin')
            sys.exit()
        return self.state_dict

if __name__ == '__main__':

    n = NMT()
    n.setup_for_interactive()
    n.loop()



