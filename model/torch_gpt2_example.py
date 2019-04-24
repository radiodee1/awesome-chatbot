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

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()


past = None

while True:
    num = 0
    text_1 = input(">>> ")
    text_2 = ""
    past = None
    decode_list = []
    while num < 10:
        #past = None

        indexed_tokens_2 = tokenizer.encode(text_1 + " . " + text_2)

        # Convert inputs to PyTorch tensors

        tokens_tensor_2 = torch.tensor([indexed_tokens_2])

        with torch.no_grad():

            predictions_1, past = model(tokens_tensor_2, past=past)

        predicted_index = torch.argmax(predictions_1[0, -1, :]).item()
        decode_list.append(predicted_index)
        predicted_token = tokenizer.decode([predicted_index])


        print(text_1 + ' - ' +  text_2.strip('\n'), '[', predicted_index,'-' +predicted_token + '-', ']')
        #if len(predicted_token.strip()) > 0 or True:
        #text_2 += " " + predicted_token

        text_2 += predicted_token

        if predicted_token.strip() in ['.','?','!'] or predicted_token[0]  in ['.','?','!']:
            #past = None
            break
        num += 1
    print(text_2)
    #print(tokenizer.decode(decode_list))