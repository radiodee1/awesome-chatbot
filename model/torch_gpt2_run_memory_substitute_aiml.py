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
sys.path.append('..')
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
from model.nmt_aiml_commands import Commands
from model.nmt_wiki_commands import Wikipedia
import aiml

realpath = os.path.dirname(os.path.realpath(__file__))
endoftext = '<|endoftext|>'

print('NOTE: consider using this source file: --source_file ../data/tf_gpt2_data/774M/converted/pytorch_model.bin')

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

        self.hp_config = None

        self.output_lang = None
        self.commands = None
        self.kernel = None
        self.wiki = None

        self.common = ''
        self.common_pre = ''
        self.common_wiki = ''
        self.previous_sentences = []
        self.sentences_formatted = ''
        self.gather_sentences = False
        self.recent_in = ''
        self.recent_text = ''
        self.save_num = 20
        self.save_on_failure = False
        self.use_common = True

        self.reply_aiml = None
        self.reply_aiml_dupes = 1
        self.token_limit = 1024

        self.q_string = ['Q: ']
        self.a_string = ['A: ']

        self.name = 'Jane'

        if True:
            self.q_string = [ 'Q: ', 'Q :', 'Q.']
            self.a_string = [ 'A: ', 'A :', self.name+':', 'A.']

    def setup_for_interactive(self):
        self.get_args()
        self.load_state_dict()
        self.load_model()

        self.commands = Commands()
        self.wiki = Wikipedia()

        ## this is not used but is required for bot software...
        self.output_lang = Lang('lang')
        for i in range(len(self.enc.encoder.items())):
            self.output_lang.addWord(self.enc.decode([i]))

        self.kernel = aiml.Kernel()
        self.kernel.verbose(False)
        self.kernel.learn("../data/std_startup.xml")

        ## do this also with each input...
        self.prepare_common()

    def prepare_common(self):
        self.common = ''
        self.common_pre = ''
        #self.common_wiki = ''
        a_chars = self.a_string[0]
        q_chars = self.q_string[0]

        now = datetime.datetime.now()
        time = now.strftime("%I:%M %p")
        date = now.strftime("%B %d, %Y")
        name = self.name
        profession = 'student'
        location = 'New York'
        #key_action_string = '\n ' + a_chars + 'play media.\n'

        self.common += ' '
        #self.common += q_chars + 'Hello?\n '

        if self.reply_aiml is None:
            #if self.common_pre == '':
            self.common_pre += a_chars + 'Hello. Hi' + '.\n '

            self.common += q_chars + 'What is your name?\n '
            self.common += a_chars + 'My name is ' + name + '.\n '
            self.common += q_chars + 'What time is it?\n '
            self.common += a_chars + 'The time is ' + time + ' ' + date + '.\n '
            #self.common += q_chars + 'What is your job?\n '
            self.common += a_chars + 'My job is as a ' + profession + '.\n '
            #self.common += q_chars + 'Where are you?\n '
            self.common += a_chars + "I am in " + location + '. \n '
        if self.reply_aiml is not None:
            self.common += '\n ' + self.reply_aiml + '\n '

    def get_sentence(self, i):

        ## aiml and rule based stuff ##
        prep_copy_boolean = False
        k = i.replace("'", '').replace('?','').replace('.','').replace('!', '')
        r = self.kernel.respond(k)
        url = self.detect_url(r)
        z = ''
        if url and self.args.apps == True:
            print(url)
            if url == self.wiki.url_search:
                self.wiki.set_topic(r[len(url):])
                z = self.wiki.get_text()
                self.common_wiki = z
            if url == self.wiki.url_stop:
                self.common_wiki = ''
                r = 'ok'
        elif url and url != self.wiki.url_stop:
            i = ''
            r = ''
        elif url and url == self.wiki.url_stop:
            i = ''
            r = 'ok'
            self.common_wiki = ''

        if r.strip() != "":
            self.reply_aiml = ''
            for _ in range(self.reply_aiml_dupes):
                #self.reply_aiml += self.q_string[0] + i + '? \n '
                self.reply_aiml += self.a_string[0] + r + '\n\n '
                #self.reply_aiml += r + '\n\n '
        else:
            self.reply_aiml = None
            prep_copy_boolean = True

        if self.use_common:
            self.recent_in = i
            i = self.q_string[0] + i + '?'

            if self.reply_aiml is None:
                s = self.sentences_formatted
            else:
                s = ''
                #i = ''

            self.prepare_common()
            if self.common_wiki != '':
                #print('here 1',i)
                self.common_pre = ''
                self.common = ''
                self.common_wiki = ' '.join(self.common_wiki.split(' ')[:self.token_limit // 2 - len(i.split(' '))]) # -(len(i.split(' ')) + 800)])
                #print(self.common_wiki, 'here 2', s)
                s = ''
                pass

            i = self.common_wiki + ' ' + self.common_pre + '\n' + s + "\n" + self.common + '\n' + i

            print('',"+" * 10, '\n', i, '\n','+' * 10)
            print(len(i.split()), 'tokens')
        i = self.prepare_input(i)

        self.args.text = i
        text = self.text_generator()
        #self.recent_text = text

        if not self.args.quiet or True: print(text)

        text = self.prepare_output(text)
        text = re.sub(endoftext, '', text)
        self.recent_text = text
        self.prep_recent(prep_copy_boolean or True)

        print(text,"<")

        ## if you want to launch apps !!
        if self.args.apps is True:
            strip = True
            if url or len(self.common_wiki) > 2:
                if url is None: url = ''
                self.recent_in = 'find ' + url
                strip = False
            elif self.commands.is_command(self.recent_in):
                self.commands.do_command(self.recent_in, strip)

        return text

    def loop(self):
        while True:
            try:
                i = input("> ")
                self.get_sentence(i)
            except EOFError:
                print()
                exit()
            except KeyboardInterrupt:
                print()
                exit()

    def prepare_input(self, i):
        self.random_seed()

        if False:
            i = self.q_string[0] + i + '?'
        else:
            i = i + "?"
        #print(i)
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

        for z in self.a_string:
            z = z.lower()
            if i.lower().startswith(z): i = i[len(z):]

        for z in self.q_string:
            z = z.lower()
            if i.lower().startswith(z): i = i[len(z):]

        #if len(i.split('.')) > 1:
        #    i = i.split('.')[0]

        if len(i.split('?')) > 1:
            i = i.split('?')[0]

        if len(i.split('!')) > 1:
            i = i.split('!')[0]

        start = i[:]
        num = 0
        default = ''
        while num < 5:

            i = start[:]
            out = []
            for ii in i.split(' '):

                out.append(ii)

                if (ii.endswith('.') or ii.endswith('!') or ii.endswith('?')) and len(ii) > 1 and ii.count('.') >= 1:
                    break
            i = ' '.join(out)

            if num == 0: default = i

            num += 1

        if i.strip() == '': i = default

        i = re.sub('[;]','',i)
        if contains_junk is True:
            i = ''

        if self.gather_sentences:
            i = i.strip()
            for z in self.q_string + self.a_string:
                z = z.lower()
                if i.lower().startswith(z): i = i[len(z):]

            i = re.sub('[?!]', ' ', i)

        ## long sentences with comma ##
        slen = self.args.length
        sout = ''
        if len(i.split(' ')) > slen // 2 and ',' in i:
            for x in i:
                if x != ',':
                    sout += x
                elif x == ',':
                    break
            i = sout
        return i

    def prep_recent(self, prep_copy_boolean=True):
        self.recent_in = self.q_string[0] + self.recent_in.strip('.').lower()
        self.recent_text = self.a_string[0] + self.recent_text.strip('.').lower()
        y = 'yes'
        n = 'no'
        for a in self.previous_sentences:
            a = a.replace('.', '')
            if (self.recent_text is not None and len(self.recent_text.split(' ')) == 1 and self.recent_text.lower() in a.lower().split(' ')):
                if y not in self.recent_text.lower() and n not in self.recent_text.lower():
                    self.recent_text = None
            if self.recent_in is not None and len(self.recent_in.split(' ')) == 1 and self.recent_in.lower() in a.lower().split(' '):
                self.recent_in = None

            if self.recent_text is not None and self.recent_text.lower().strip() == a.lower().strip():
                if y not in self.recent_text.lower() and n not in self.recent_text.lower():
                    self.recent_text = None
            if self.recent_in is not None and self.recent_in.lower().strip() == a.lower().strip():
                self.recent_in = None

        if not prep_copy_boolean:
            self.recent_in = None
            self.recent_text = None

        if self.recent_in is not None and self.recent_text is not None and 'time' not in self.recent_in and 'name' not in self.recent_in:
            self.previous_sentences.extend([self.recent_in, self.recent_text])


        if self.save_num > -1:
            self.previous_sentences = self.previous_sentences[-self.save_num:]

        #print(self.previous_sentences)
        s = ''
        for k in self.previous_sentences:
            k = k.strip().strip('.').strip('\n')
            for z in self.a_string + self.q_string:
                z = z.lower()
                if k.lower().startswith(z) and False: k = k[len(z):]
            if len(k) > 0:
                s += k + '.\n'
        #s = ['---'] + s + ['---']
        self.sentences_formatted = s

    def detect_url(self, txt):
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', txt)
        print(len(urls), 'urls')
        if len(urls) > 0:
            return urls[0]
        else:
            return None
    #########################################

    def random_seed(self):
        seed = random.randint(0, 2147483647)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        pass

    def get_encoder(self):
        print(self.args.source_file)
        source_path = self.args.source_file.split('/')[:-1]
        source_path = '/'.join(source_path) + '/'
        print(source_path)
        with open(realpath + '/' + source_path + '/encoder.json', 'r') as f:
            encoder = json.load(f)
        with open(realpath + '/' + source_path + '/vocab.bpe', 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        return Encoder(
            encoder=encoder,
            bpe_merges=bpe_merges,
        )

    def get_config(self):
        print(self.args.source_file)
        source_path = self.args.source_file.split('/')[:-1]
        source_path = '/'.join(source_path) + '/'
        print(source_path)
        if '774M' in source_path :
            print('774M', 'model specific configs')
            #self.use_common = False
            self.args.temperature = 1e-10
            self.args.top_k = 100
        if os.path.isfile(realpath + '/' + source_path + '/config.json'):
            with open(realpath + '/' + source_path + '/config.json', 'r') as f:
                hp_config = json.load(f)
                print(hp_config)
                self.config = GPT2Config(
                    vocab_size_or_config_json_file=hp_config['vocab_size'],
                    n_embd=hp_config['n_embd'],
                    n_layer=hp_config['n_layer'],
                    n_head=hp_config['n_head'],
                    # intermediate_size=self.intermediate_size,
                    # hidden_act=self.hidden_act,
                    # hidden_dropout_prob=self.hidden_dropout_prob,
                    # attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                    n_positions=hp_config['n_positions'],
                    n_ctx=hp_config['n_ctx']
                    # type_vocab_size=self.type_vocab_size,
                    # initializer_range=self.initializer_range
                )
        print(self.config)

    def get_args(self ):
        parser = argparse.ArgumentParser()
        parser.add_argument("--text", type=str, required=False)
        parser.add_argument("--quiet", type=bool, default=True)
        parser.add_argument("--nsamples", type=int, default=1)
        parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
        parser.add_argument("--batch_size", type=int, default=-1)
        parser.add_argument("--length", type=int, default=25)
        parser.add_argument("--temperature", type=float, default=0.0001)
        parser.add_argument("--top_k", type=int, default=40)
        parser.add_argument("--apps", type=bool, required=False, default=False)
        parser.add_argument("--source_file", type=str, required=False, default='torch_gpt2/GPT2/gpt2-pytorch_model.bin')
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

        self.get_config()

        # Load Model
        self.enc = self.get_encoder()
        if self.config is None: self.config = GPT2Config()
        self.model = GPT2LMHeadModel(self.config)
        self.model = load_weight(self.model, self.state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(self.config)

    def text_generator(self):

        if self.args.length == -1:
            self.args.length = self.config.n_ctx // 2
        elif self.args.length > self.config.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % self.config.n_ctx)

        if self.args.quiet is False: print(self.args.text)
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
        print(self.args.source_file)
        source_path = self.args.source_file.split('/')[:-1]
        source_path = '/'.join(source_path) + '/'
        print(source_path, 2)

        p = realpath + '/' + self.args.source_file #'./torch_gpt2/gpt2-pytorch_model.bin'

        #p = realpath + '/' + source_path + '/' + self.args.source_file

        print(p)
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



