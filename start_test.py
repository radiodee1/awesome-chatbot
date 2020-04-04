#!/usr/bin/python3
# Requires PyAudio and PySpeech.
from __future__ import division, print_function

try:
    import re
    import sys

    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    import pyaudio
    from six.moves import queue
except:
    pass
finally:
    print('google cloud load')