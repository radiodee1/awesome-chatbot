#!/usr/bin/python3

from __future__ import unicode_literals, print_function, division
mode = 'zero' #'sequence' # 'gpt2' 'zero'
speech_start = 'hello'
must_stop = True
no_tokenize_weak = False
pin_skip = False

sound_tones = ['sequence', 'signal', 'wiki', 'transformer']

import sys
import os
sys.path.append('..')
sys.path.append(os.path.abspath('../model/'))
sys.path.append(os.path.abspath('../seq_2_seq/'))
sys.path.append(os.path.abspath('../transformer/'))

csv_label = 't2t'
csv_voc_tot = 0

mode = str(os.environ['CHATBOT_MODE'])
if os.environ['CHATBOT_START']:
    speech_start = str(os.environ['CHATBOT_START'])
do_not_end = True
stat_limit = int(str(os.environ['STAT_LIMIT']))

stat_enum = int(str(os.environ['STAT_ENUM'])) ## could be zero
stat_tab = int(str(os.environ['STAT_TAB'])) ## could be very large

if mode == 'sequence':
    #import seq_2_seq.seq_2_seq as model
    import seq_2_seq.seq_2_seq_tutorial as model
    import seq_2_seq.tokenize_weak as tokenize_weak
    csv_label = 'gru'
    csv_voc_tot = 7826


elif mode == 'memory' or mode == 'signal':
    sys.path.append(os.path.abspath('../model/torch_gpt2/'))
    import model.torch_gpt2_run_memory_substitute_aiml_sm as model
    import model.tokenize_weak as tokenize_weak
    csv_label = 'gpt'
    csv_voc_tot = 50000

elif mode == 'wiki':
    sys.path.append(os.path.abspath('../model/torch_gpt2/'))
    import model.torch_gpt2_run_memory_substitute_aiml_lrg as model
    import model.tokenize_weak as tokenize_weak
    must_stop = False
    no_tokenize_weak = True
    mode = 'signal'
    csv_label = 'gpt'
    csv_voc_tot = 50000

elif mode == 'transformer':
    os.chdir('../transformer/')
    import transformer.tf_t2t_train_run as model
    #import model.torch_gpt2_run_memory as model
    import model.tokenize_weak as tokenize_weak
    must_stop = False
    mode = 'signal'
    csv_label = 't2t'
    csv_voc_tot = 8170

if mode in sound_tones:
    mode = 'signal'

#import bot.game_sr as sr
#import bot.game_voice as v
#import model.tokenize_weak as tokenize_weak
import model.settings as settings
import argparse
import time

try:
    import RPi.GPIO as GPIO
    led_pin_a = 12
    led_pin_b = 16
    print('load gpio')
except:
    pin_skip = True
    print('no load gpio')


base_filename = ''

class Game:
    def __init__(self):
        global base_filename
        #self.pin_setup()

        #self.pin_both()

        self.model = model.NMT()
        self.model.setup_for_interactive()

        self.responses = {}
        self.responses_list = []
        self.responses_words = {}
        self.responses_words_list = []

        self.csv_responses = []
        self.csv_original = []
        self.csv_repeats = []
        self.csv_total = []

        self.csv_words = []
        self.csv_words_original = []
        self.csv_words_repeats = []
        self.csv_words_total = []

        self.chart = {}
        #self.voice = v.VoiceOut()
        #self.sr = sr.VoiceGoogleSR()

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
        #self.voice.beep_out()
        #self.voice.speech_out(speech_start)
        self.time_total = 5
        self.time_allowed = 3.5
        if mode == 'sequential': self.time_allowed = 100


    def loop(self):
        global mode
        count = 0
        num = 0
        print('starting')
        self.print_contents(pr=False, code='w')
        while True:
            #self.pin_a_on()
            #i = self.sr.voice_detection()
            if stat_tab <= num: break
            i = input(str(num)+ '> ')
            #print(i)
            #self.pin_a_off()
            if not no_tokenize_weak: i = tokenize_weak.format(i)
            if (self.compare_sentence_to_list(i, self.words_start) and count <= 0) or self.first_run:
                count = self.count_max

                if not self.first_run:
                    #self.voice.speech_out('yes')
                    i = ''
                self.first_run = False
            if self.compare_sentence_to_list(i, self.words_stop) and must_stop:
                count = 0
                #print('stopping')
            i = self.check_sentence(i)
            print(i)
            if len(i) > 0:
                if count > 0 :
                    #if mode == 'signal': self.voice.beep_out()
                    ts = time.time()
                    if (i.strip() == '' or len(i.strip()) == 0) or i.strip() == "'" :
                        i = '.'
                    out = self.model.get_sentence(i)
                    te = time.time()
                    #if mode == 'signal': self.voice.beep_out()
                    if out not in self.responses:
                        self.responses[out] = 1
                    else:
                        self.responses[out] += 1
                    for word in out.split():
                        if word not in self.responses_words:
                            self.responses_words[word] = 1
                        else:
                            self.responses_words[word] += 1
                        self.responses_words_list.append(word) ## do not use !!
                        pass
                    #print(' count:', self.responses[out])
                    ## seconds ##
                    self.time_total = (te - ts)
                    if self.time_total > self.time_allowed: mode = 'signal'
                    #print(self.time_total, 'time')

                    blacklisted = False
                    for jj in self.blacklist:
                        if out.startswith(jj):
                            blacklisted = True
                    if not blacklisted:
                        print('[',out, '] count:', self.responses[out])
                        self.responses_list.append([i, out])
                        #self.voice.speech_out(out)
            if not do_not_end: count -= 1
            num += 1
            if num % 100 == 0 and num <= stat_limit:
                self.print_contents(pr=False, code='a')
            if count <= 0 :
                #print('quiet')
                pass

    def check_sentence(self, i):
        i = i.split(' ')
        out = []
        for ii in i:
            if ii == 'eol':
                continue
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

    def print_contents(self, pr=False, code='w'):
        if pr is True: print('\n-----')
        with open('../saved/output.'+csv_label+'.txt',code) as f:
            if pr is True:
                for i in self.responses_list:
                    print(i)
                    f.write(i[0] + ' ' + i[1] + '\n')
                print('-----')
                f.write('-----\n')
            z = [sorted(self.responses.items(), key=lambda kv:(kv[1], kv[0]),reverse=True)]
            #print(z)
            num = 0
            original = 0
            for key in z[0]:
                if pr is True:
                    print(num, key)
                    f.write(str(num) +' ' + str(key[0]) + ' [' + str(key[1]) + '] \n')
                num += 1
                if key[1] == 1: original += 1
            print('-----')
            f.write('-----\n')
            f.write(str(len(self.responses)) + ' responses / ' + str(len(self.responses_list)) + ' questions \n')
            f.write(str(original) + ' original / ' + str(len(self.responses_list)) + ' questions \n')
            f.write(str(len(self.responses) - original) + ' repeats / ' + str(len(self.responses_list)) + ' questions \n')
            #f.write(str(len(self.responses_words)) + ' words \n')
            print(str(len(self.responses)) + ' responses / ' + str(len(self.responses_list)) + ' questions')
            print(original, 'original /', str(len(self.responses_list)), 'questions')
            print(len(self.responses) - original, 'repeats /', str(len(self.responses_list)), 'questions')

            ### just words ###
            z = [sorted(self.responses_words.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]

            num_words = 0
            original_words = 0
            for key in z[0]:
                num_words += 1
                if key[1] == 1: original_words += 1

            f.write(str(len(self.responses_words)) + ' responses / ' + str(len(self.responses_words_list)) + ' words \n')
            f.write(str(original_words) + ' original / ' + str(len(self.responses_words_list)) + ' words \n')
            f.write(str(len(self.responses_words) - original_words) + ' repeats / ' + str(len(self.responses_words_list)) + ' words \n')
            f.write(str(len(self.responses_words)) + ' words \n')
            print(str(len(self.responses_words)) + ' responses / ' + str(len(self.responses_words_list)) + ' words')
            print(original_words, 'original /', str(len(self.responses_words_list)), 'words')
            print(len(self.responses_words) - original_words, 'repeats /', str(len(self.responses_words_list)), 'words')
            #print(len(self.responses_words), 'words')

        self.csv_responses.append(len(self.responses))
        self.csv_original.append(int(original))
        self.csv_repeats.append(len(self.responses) - original)
        self.csv_total.append(len(self.responses_list)) ## num of sentences

        self.csv_words.append(len(self.responses_words))
        self.csv_words_original.append(int(original_words))
        self.csv_words_repeats.append(len(self.responses_words) - original_words)
        self.csv_words_total.append(len(self.responses_list)) ## num of sentences

        with open('../saved/output.'+csv_label+'.csv', 'w') as g:
            g.write('Sentences.'+csv_label+',')
            for i in self.csv_total:
                g.write(str(i) + ',')
            g.write('\n')
            g.write('Repeated_Sentences.'+csv_label+',')
            for i in self.csv_repeats:
                g.write(str(i) + ',')
            g.write('\n')
            g.write('Sentences_Used_Once.'+csv_label+',')
            for i in self.csv_original:
                g.write(str(i) + ',')
            g.write('\n')
            g.write('Total_Sentences.'+csv_label+',')
            for i in self.csv_responses:
                g.write(str(i) + ',')
            g.write('\n')

            g.write('Repeated_Words.'+csv_label+',')
            for i in self.csv_words_repeats:
                g.write(str(i) + ',')
            g.write('\n')
            g.write('Words_Used_Once.'+csv_label+',')
            for i in self.csv_words_original:
                g.write(str(i) + ',')
            g.write('\n')
            g.write('Total_Word_Responses.'+csv_label+',')
            for i in self.csv_words:
                g.write(str(i) + ',')
            g.write('\n')
            g.write('Total_Voc.' + csv_label + ',')
            for i in self.csv_words:
                g.write(str(csv_voc_tot) + ',')
            g.write('\n')

    def print_tab_file(self, pr=False, code='w'):
        if pr is True: print('\n-----')
        if stat_enum > 0:
            with open('../saved/output.'+csv_label+'.enu.txt',code) as f:
                count = self.responses
                z = self.responses_list
                z = sorted(self.responses_list, key=lambda kv: (kv[1], kv[0]), reverse=True)

                self.chart = {}
                num = 0
                for key in z:
                    #print(key[1], key[0], num)
                    if count[key[1]] > 1 and key[1] not in self.chart and num < stat_enum:
                        self.chart[key[1]] = num
                        f.write(str(key[1]) + '\t' + str(self.chart[key[1]]) + '\n')
                        num += 1
                    pass
        else:
            with open('../saved/output.'+ csv_label + '.enu.txt', 'r') as f:
                z = f.readlines()
                self.chart = {}
                for line in z:
                    key = line.split('\t')
                    self.chart[key[0]] = key[-1]

        with open('../saved/output.'+csv_label+'.tab.txt',code) as f:
            count = self.responses
            z = self.responses_list
            l = []
            num = 0
            original = 0
            for key in z:
                if pr is True:
                    print(num, key)

                    if count[key[1]] > 1 and key[1] in self.chart and key[1] not in l:
                        f.write(str(key[0]) + '\t'+ key[1] + '\t' + str(count[str(key[1])]) + '\t' + str(self.chart[key[1]]) + '\n')
                        l.append(key[0])
                num += 1
                if key[1] == 1: original += 1
            print('-----')
        pass


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
    try:
        g.loop()
    except EOFError:
        pass
    finally:
        #if g.model.voc.num_words is not None:
        #    print(g.model.voc.num_words, 'voc')
        g.print_tab_file(pr=True, code='w')
        pass
