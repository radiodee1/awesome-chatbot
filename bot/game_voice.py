#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function
import os
from gtts import gTTS
from io import BytesIO
import pygame
from queue import Queue
import sys
sys.path.append('..')
from model.settings import hparams

class VoiceOut:
    def __init__(self):
        self.dir_out = hparams['data_dir']
        self.use_me = Queue()
        self._do_quit = False
        pass

    def speech_out(self,text=""):
        if not self.use_me.empty() :
            return
        self.use_me.put(True)
        if len(text) > 0 and text.split(' ')[0] not in ['.','!','?',',']:
            try:
                #mp3_fp = BytesIO()
                tts = gTTS(text=text, lang='en', slow=False, lang_check=False)
                #tts.write_to_fp(mp3_fp)
                #path = os.path.join(self.dir_out,"temp_speech.mp3")
                #tts.save(path)
            except AssertionError:
                print('assertion error.')
                pass
            except e:
                print(e)
                pass
            pass
            #os.system("mpg123 " + path + " > /dev/null 2>&1 ")
            try:
                with BytesIO() as f:

                    pygame.mixer.init(24000, -16, 1, 4096)

                    tts.write_to_fp(f)
                    f.seek(0)
                    pygame.mixer.music.load(f)
                    pygame.mixer.music.play()
                    num = 0
                    while pygame.mixer.music.get_busy() and num < 100:
                        pygame.time.Clock().tick(10)
                        num += 1
                        print(num)
                    if self._do_quit: pygame.quit()
            except e:
                print(e)
                pass

        while not self.use_me.empty(): self.use_me.get()
    pass

    def beep_out(self):
        if not self.use_me.empty() :
            return
        self.use_me.put(True)
        try:
            path = os.path.join(self.dir_out,"beep.mp3")
            pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(0.7)
            pygame.mixer.music.play()
            num = 0
            while pygame.mixer.music.get_busy() and num < 100:
                pygame.time.Clock().tick(10)
                num += 1
                #print(num)
            if self._do_quit: pygame.quit()
        except e:
            print(e)
            pass
        while not self.use_me.empty(): self.use_me.get()

        #os.system("mpg123 " + path + " > /dev/null 2>&1 ")

if __name__ == '__main__':
    v = VoiceOut()
    v.speech_out("hello")
    v.speech_out("this is a test")
    v.speech_out("goodbye")
    v.beep_out()
