#!/usr/bin/python3.6

#@article{radford2019language,
#  title={Language Models are Unsupervised Multitask Learners},
#  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
#  year={2019}
#}

import torch
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

class GPT2_small:
    def __init__(self):

        self.tokenizer = None
        self.model = None
        self.wordcount = 15
        self.words_end = ['.', '?', '!']

        self.past = None

    def setup_for_interactive(self):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Load pre-trained model (weights)
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()

    def get_sentence(self, i):
        num = 0
        text_1 = i
        text_2 = ""
        self.past = None
        decode_list = []
        while num < self.wordcount:

            indexed_tokens_2 = self.tokenizer.encode(text_1 + " . " + text_2)
            tokens_tensor_2 = torch.tensor([indexed_tokens_2])

            with torch.no_grad():
                predictions_1, self.past = self.model(tokens_tensor_2, past=self.past)

            predicted_index = torch.argmax(predictions_1[0, -1, :]).item()
            decode_list.append(predicted_index)
            predicted_token = self.tokenizer.decode([predicted_index])

            print(text_1 + ' - ' + text_2.strip('\n'), '[', predicted_index, '-' + predicted_token + '-', ']')

            text_2 += predicted_token

            if predicted_token.strip() in self.words_end or predicted_token[0] in self.words_end:
                # past = None
                break
            num += 1
        print(text_2)
        return text_2


    def loop(self):

        self.past = None

        while True:
            #num = 0
            text_1 = input(">>> ")
            self.get_sentence(text_1)


if __name__ == '__main__':
    g = GPT2_small()
    g.setup_for_interactive()
    g.loop()