#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
import os
from gtts import gTTS
import sys
sys.path.append('..')
from model.settings import hparams

class VoiceOut:
    def __init__(self):
        self.dir_out = hparams['data_dir']
        pass

    def speech_out(self,text=""):
        if len(text) > 0:
            tts = gTTS(text=text, lang='en')
            path = os.path.join(self.dir_out,"temp_speech.mp3")
            tts.save(path)
            os.system("mpg321 " + path + " > /dev/null 2>&1 ")
        pass

if __name__ == '__main__':
    v = VoiceOut()
    v.speech_out("hello")
    v.speech_out("this is a test")
    v.speech_out("goodbye")
