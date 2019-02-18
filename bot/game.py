#!/usr/bin/python3

from __future__ import unicode_literals, print_function, division

import sys
import os
sys.path.append('..')
sys.path.append(os.path.abspath('../model/'))

import model.seq_2_seq_beam as model
import game_sr as sr
import game_voice as v
import model.tokenize_weak as tokenize_weak
import model.settings as settings
import argparse
import time

base_filename = ''

class Game:
    def __init__(self):
        global base_filename

        self.model = model.NMT()
        self.model.setup_for_interactive()

        self.voice = v.VoiceOut()
        self.sr = sr.VoiceGoogleSR()

        self.words_name = ['chatbot','mutter','robot']
        self.words_start = ['start','talk','answer','reply'] # talk answer reply
        self.words_stop = ['stop','exit','quit','quiet','silence']
        self.words_start += self.words_name
        self.count_max = 5

        self.blacklist = [
            "i don't know",
            "i do not know"
        ]

        '''
        self.time_start = 0
        self.time_end = 0
        self.time_in_seconds = 5
        '''

    def loop(self):
        count = 0
        while True:
            i = self.sr.voice_detection()
            i = tokenize_weak.format(i)
            if self.compare_sentence_to_list(i, self.words_start):
                count = self.count_max
                self.voice.speech_out('yes')
                print('starting')
            if self.compare_sentence_to_list(i, self.words_stop):
                count = 0
                print('stopping')
            i = self.check_sentence(i)
            if len(i) > 0:
                if count > 0 :
                    out = self.model.get_sentence(i)
                    xx = []
                    for i in out.split():
                        if i != settings.hparams['unk']:
                            xx.append(i)
                    out = ' '.join(xx)
                    print(out)
                    blacklisted = False
                    for jj in self.blacklist:
                        if out.startswith(jj):
                            blacklisted = True
                    if not blacklisted:
                        self.voice.speech_out(out)
            count -= 1
            if count <= 0 :
                print('quiet')

    def check_sentence(self, i):
        i = i.split(' ')
        out = []
        for ii in i:
            if ii in self.model.output_lang.word2index:
                out.append(ii)
        return ' '.join(out)

    def compare_sentence_to_list(self, sent, l):
        ## look for single word ##

        i = sent.split(' ')
        for ii in i:
            if ii in l:
                return True
        return False



if __name__ == '__main__':
    '''
    print('enter one file path for input of saved weights')
    #parser = argparse.ArgumentParser(description='Chatbot game program.')
    #parser.add_argument('weights', help='file for saved weights and biases.')

    #args = parser.parse_args()
    #args = vars(args)
    #name = str(args['weights'])
    print(sys.argv)
    name = str(sys.argv[1])
    print(name)
    name = name.split('/')[-1]
    name = name.split('.')[0]
    base_filename = name
    '''
    print('enter command line options for NMT class')
    g = Game()
    g.loop()