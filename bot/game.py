#!/usr/bin/python3

from __future__ import unicode_literals, print_function, division

import sys
import os
sys.path.append('..')
sys.path.append(os.path.abspath('../model/'))

import model.seq_2_seq as model
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
        self.words_start = ['start','talk','answer','reply','hello','hi']
        self.words_stop = ['stop','exit','quit','quiet','silence']
        self.words_start += self.words_name
        self.count_max = 15

        self.blacklist = [
            #"i don't know",
            #"i do not know"
        ]
        self.voice.beep_out()


    def loop(self):
        count = 0
        while True:
            i = self.sr.voice_detection()
            i = tokenize_weak.format(i)
            if self.compare_sentence_to_list(i, self.words_start) and count <= 0:
                count = self.count_max
                self.voice.speech_out('yes')
                print('starting')
                i = ''
            if self.compare_sentence_to_list(i, self.words_stop):
                count = 0
                print('stopping')
            i = self.check_sentence(i)
            if len(i) > 0:
                if count > 0 :
                    out = self.model.get_sentence(i)
                    
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

    print('enter command line options for NMT class')
    g = Game()
    g.loop()