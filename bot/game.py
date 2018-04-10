#!/usr/bin/python3

from __future__ import unicode_literals, print_function, division

import sys
import os
sys.path.append('..')
sys.path.append(os.path.abspath('../model/'))

import model.pytorch as model
import game_sr as sr
import game_voice as v
import model.tokenize_weak as tokenize_weak


class Game:
    def __init__(self):
        self.model = model.NMT()
        self.model.setup_for_interactive()

        self.voice = v.VoiceOut()
        self.sr = sr.VoiceGoogleSR()


    def loop(self):
        while True:
            #i = input('>')
            i = self.sr.voice_detection()
            i = tokenize_weak.format(i)
            out = self.model.get_sentence(i)
            print(out)
            self.voice.speech_out(out)

if __name__ == '__main__':
    g = Game()
    g.loop()