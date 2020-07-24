#!/usr/bin/env python3

from __future__ import unicode_literals, print_function, division
mode = 'zero' #'sequence' # 'gpt2' 'zero'
speech_start = 'hello'
must_stop = True
always_beep = True
no_tokenize_weak = False
pin_skip = False

sound_tones = ['sequence', 'signal', 'wiki', 'transformer']

import sys
import os
sys.path.append('..')
sys.path.append(os.path.abspath('../model/'))
sys.path.append(os.path.abspath('../seq_2_seq/'))
sys.path.append(os.path.abspath('../transformer/'))

mode = str(os.environ['CHATBOT_MODE'])
if os.environ['CHATBOT_START']:
    speech_start = str(os.environ['CHATBOT_START'])
do_not_end = True

if mode == 'sequence':
    #import seq_2_seq.seq_2_seq as model
    import seq_2_seq.seq_2_seq_tutorial as model
    import seq_2_seq.tokenize_weak as tokenize_weak


elif mode == 'memory' or mode == 'signal':
    sys.path.append(os.path.abspath('../model/torch_gpt2/'))
    import model.torch_gpt2_run_memory_substitute_aiml_sm as model
    import model.tokenize_weak as tokenize_weak

elif mode == 'wiki':
    sys.path.append(os.path.abspath('../model/torch_gpt2/'))
    import model.torch_gpt2_run_memory_substitute_aiml_lrg as model
    import model.tokenize_weak as tokenize_weak
    must_stop = False
    no_tokenize_weak = True
    mode = 'signal'

elif mode == 'transformer':
    os.chdir('../transformer/')
    import transformer.tf_t2t_train_run as model
    #import model.torch_gpt2_run_memory as model
    import model.tokenize_weak as tokenize_weak
    must_stop = False
    mode = 'signal'

if mode in sound_tones:
    mode = 'signal'

import bot.game_sr as sr
import bot.game_voice as v
#import model.tokenize_weak as tokenize_weak
import model.settings as settings
import argparse
import time

try:
    import RPi.GPIO as GPIO
    led_pin_a = 12
    led_pin_b = 16
    print('load rpi gpio')
except:
    pin_skip = True
    try:
        import Jetson.GPIO as GPIO
        pin_skip = False
        led_pin_a = 12
        led_pin_b = 16
        print('load jetson gpio')
    except:
        pin_skip = True
        print('no load gpio')


base_filename = ''

class Game:
    def __init__(self):
        global base_filename
        self.pin_setup()

        self.pin_both()

        self.model = model.NMT()
        self.model.setup_for_interactive()

        self.voice = v.VoiceOut()
        self.sr = sr.VoiceGoogleSR()

        self.words_name = ['chatbot','mutter','robot']
        self.words_start = ['start','talk','answer','reply','hello','hi','okay']
        self.words_stop = ['stop','exit','quit','quiet','silence']
        self.words_start += self.words_name
        self.count_max = 15

        self.first_run = True

        self.blacklist = [
            #"i don't know",
            #"i do not know"
        ]
        self.voice.beep_out()
        self.voice.speech_out(speech_start)
        self.time_total = 5
        self.time_allowed = 3.5
        if mode == 'sequential': self.time_allowed = 100


    def loop(self):
        global mode
        count = 0
        while True:
            self.pin_a_on()
            i = self.sr.voice_detection()
            self.pin_a_off()
            if not no_tokenize_weak: i = tokenize_weak.format(i)
            if (self.compare_sentence_to_list(i, self.words_start) and count <= 0) or self.first_run:
                count = self.count_max
                print('starting')
                if not self.first_run:
                    self.voice.speech_out('yes')
                    i = ''
                self.first_run = False
            if self.compare_sentence_to_list(i, self.words_stop) and must_stop:
                count = 0
                print('stopping')
            i = self.check_sentence(i)
            if len(i) > 0:
                if count > 0 :
                    if mode == 'signal': self.voice.beep_out()
                    ts = time.time()
                    out = self.model.get_sentence(i)
                    te = time.time()
                    if mode == 'signal': self.voice.beep_out()

                    ## seconds ##
                    self.time_total = (te - ts)
                    if self.time_total > self.time_allowed or always_beep: mode = 'signal'
                    print(self.time_total, 'time')

                    blacklisted = False
                    for jj in self.blacklist:
                        if out.startswith(jj):
                            blacklisted = True
                    if not blacklisted:
                        print(out)
                        self.voice.speech_out(out)
            if not do_not_end: count -= 1
            if count <= 0 :
                print('quiet')

    def check_sentence(self, i):
        i = i.split(' ')
        out = []
        for ii in i:
            if mode != "sequence" or ii in self.model.output_lang.word2index:
                out.append(ii)
        return ' '.join(out)

    def compare_sentence_to_list(self, sent, l):
        ## look for single word ##

        i = sent.split(' ')
        for ii in i:
            if ii in l:
                return True
        return False

    def pin_setup(self):
        if pin_skip: return
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(led_pin_a, GPIO.OUT)
        GPIO.setup(led_pin_b, GPIO.OUT)

    def pin_a_on(self):
        if pin_skip: return
        GPIO.output(led_pin_a, GPIO.HIGH)
        GPIO.output(led_pin_b, GPIO.LOW)

    def pin_a_off(self):
        if pin_skip: return
        GPIO.output(led_pin_a, GPIO.LOW)
        GPIO.output(led_pin_b, GPIO.HIGH)

    def pin_both(self):
        if pin_skip: return
        GPIO.output(led_pin_a, GPIO.HIGH)
        GPIO.output(led_pin_b, GPIO.HIGH)

if __name__ == '__main__':

    print('enter command line options for NMT class')
    g = Game()
    g.loop()