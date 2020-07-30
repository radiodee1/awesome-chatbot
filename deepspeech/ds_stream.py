from deepspeech import Model
import numpy as np
import speech_recognition as sr
import argparse

sample_rate = 16000
beam_width = 500
lm_alpha = 0.75
lm_beta = 1.85
n_features = 26
n_context = 9

models_folder = 'deepspeech-0.6.0-models/'
model_name = models_folder+"output_graph.pbmm"
alphabet = models_folder+"alphabet.txt"
language_model = models_folder+"lm.binary"
trie = models_folder+"trie"

parser = argparse.ArgumentParser(description='Running DeepSpeech inference.')
parser.add_argument('--model', required=True,
                    help='Path to the model (protocol buffer binary file)')
args = parser.parse_args()


ds = Model(args.model)
#ds.enableDecoderWithLM(language_model, trie, lm_alpha, lm_beta)

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say Something")
    audio = r.listen(source)
    #fs = audio.sample_rate
    #audio = np.frombuffer(audio.frame_data, np.int16)
    print('ans:', r.recognize(audio))
    #print(ds.stt(audio))

