#!/usr/bin/env python3

import aiml as aiml_std
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import xml.etree.ElementTree as ET

class Kernel:

    def __init__(self):
        self.filename = 'name'
        self.verbose_response = True
        self.output = ""
        self.kernel = aiml_std.Kernel()
        self.tree = None
        self.root = None
        self.l = []
        self.score = []

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

    def verbose(self, isverbose):
        #print(isverbose)
        self.verbose_response = isverbose
        self.kernel.verbose(isverbose)

    def learn(self, file):
        self.filename = file
        self.kernel.learn(file)
        self.l = []
        self.score = []
        self.tree = ET.parse(file)
        self.root = self.tree.getroot()
        num = 0
        for child in self.root.iter('category'):
            pat = None
            tem = None
            for i in child:
                #print(i.tag)
                z = ''
                for c in i:
                    z = c.tag
                if self.verbose_response: print(num,z, i.tag)
                if i.tag == 'pattern':
                    pat = i.text.strip()
                    if '*' in pat:
                        z = '*'
                if i.tag == 'template':
                    tem = i.text.strip()
            if len(z) == 0: self.l.append([pat,tem])

            num += 1
            pass
        if self.verbose_response:
            print(self.l)
            print(len(self.l), num)

    def respond(self, input):
        self.score = []
        tempout = self.kernel.respond(input)
        ## checkout input and response ##
        self.output = tempout

        if len(tempout) > 0:
            return self.output
        ## compare all aiml input patttern ##
        num = 0
        for i in self.l:
            ii = i[0]
            s = self.bert_compare(ii, input)
            self.score.append(s)
            if self.verbose_response: print(num, s)
            num += 1
        ## find highest entry ##
        high = 0
        pat = ''
        num = 0
        index = -1
        for i in self.score:
            if i > high:
                high = i
                pat = self.l[num][0]
                index = num
            num += 1
        if len(pat) > 0:
            if self.verbose_response: print(input,'--' ,index, '-- find k response for --', pat)
            self.output = self.kernel.respond(pat)
        if len(self.output) is 0 and index is not -1:
            if self.verbose_response: print(input,'--' ,index, '-- print template --', self.l[index][1])
            self.output = self.l[index][1]
        return self.output

    def bert_compare(self, prompt1, prompt2):
        encoding = self.tokenizer(prompt1, prompt2, return_tensors='pt')
        loss, logits = self.model(**encoding, next_sentence_label=torch.LongTensor([1]))
        s = logits[0][0].item()
        return s

if __name__ == '__main__':

    k = Kernel()
    k.verbose(True)
    k.learn('startup.xml')
    while True:
        print(k.respond(input('> ')))