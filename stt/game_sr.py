#!/usr/bin/env python3
# Requires PyAudio and PySpeech.
from __future__ import division, print_function

import os
import re
import sys

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import pyaudio
from six.moves import queue

from pocketsphinx import LiveSpeech, get_model_path

model_path = get_model_path()

# Audio recording parameters
RATE = 44100 #16000
# note: 44100 works with device on raspberry pi
# Bus 001 Device 005: ID 0d8c:013c C-Media Electronics, Inc. CM108 Audio Controller

CHUNK = int(RATE / 400)  # 100ms... 10 for pi boards


class VoiceGoogleSR:

    def __init__(self):
        pass

    def listen_print_loop(self, responses):
        pass

    def listen(self, responses):
        pass


    def voice_detection(self):
        speech = LiveSpeech(
            verbose=False,
            sampling_rate=RATE,
            buffer_size=2048,
            no_search=False,
            full_utt=False,
            hmm=os.path.join(model_path, 'en-us'),
            lm=os.path.join(model_path, 'en-us.lm.bin'),
            dic=os.path.join(model_path, 'cmudict-en-us.dict'),
            #silence_limit= 1,
            #prev_audio=0.5,
            #threshold=4500
        )
        speech.silence_limit = 1
        speech.prev_audio=0.5
        speech.threshold = 4500

        out = ''
        for z in speech:
            out = z
            break
        return str(out)


    def run_recognition(self):
        pass

if __name__ == '__main__':
    print(model_path)

    v = VoiceGoogleSR()
    for i in range(300):  # test three iterations
        words = v.voice_detection()
        print(words)
