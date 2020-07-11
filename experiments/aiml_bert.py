#!/usr/bin/env python3

import aiml as aiml_std
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import xml.etree.ElementTree as ET
import re

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
        self.memory = {}

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
            #pat = None
            #tem = None
            pat_dict = self.pattern_factory(child)
            pat_dict['index'] = num
            '''
            for i in child:
                #print(i.tag)
                #z = ''
                #for c in i:
                #    z = c.tag
                #if self.verbose_response: print(num,z, i.tag)
                if i.tag == 'pattern':
                    pat = i.text.strip()
                    if '*' in pat:
                        z = '*'
                if i.tag == 'template':
                    tem = i.text.strip()
            '''
            self.l.append([None, None, pat_dict])

            num += 1
            pass
        if self.verbose_response:
            print(self.l)
            print(len(self.l), num)

    def respond(self, input):
        self.score = []
        tempout = '' #self.kernel.respond(input)
        ## checkout input and response ##
        self.output = tempout

        if len(tempout) > 0:
            return self.output
        ## compare all aiml input patttern ##
        num = 0
        for i in self.l:
            ii = i[0]
            input_02 = self.mod_input(i[2], input)
            ii = i[2]['pattern']
            s = self.bert_compare(ii, input_02)
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
        ## update dictionary ##
        self.mod_dict_out(self.l[index][2], input)
        print(self.l)
        print(self.memory,'<<')
        if False: #len(pat) > 0:
            if self.verbose_response: print(input,'--' ,index, '-- find k response for --', pat)
            self.output = self.kernel.respond(pat)
        if len(self.output) is 0 and index is not -1:
            if self.verbose_response: print(input,'--' ,index, '-- print template --', self.l[index][1])
            self.output = self.l[index][2]['template']
            print (ET.tostring(self.output), "???")
            self.output = self.output.text.strip()
        return self.output

    def bert_compare(self, prompt1, prompt2):
        encoding = self.tokenizer(prompt1, prompt2, return_tensors='pt')
        loss, logits = self.model(**encoding, next_sentence_label=torch.LongTensor([1]))
        s = logits[0][0].item()
        return s

    def pattern_factory(self, category):
        pat = None
        tem = None
        set = None
        get = None
        z = ''
        for i in category:
            print (i.tag, i, 'iiii')
            if i.tag == 'pattern':
                pat = ET.tostring(i)
                if '*' in pat.decode('utf-8'):
                    pat = re.sub('\*', '<star/>', pat.decode('utf-8'))
                    print (pat)

            if i.tag == 'template':
                tem = i #.text.strip()
                tem_02 = ET.tostring(i).decode('utf-8').strip()
                tem_02 = self.strip_right_left('template', tem_02)
                if '<' in tem_02 or '>' in tem_02:
                    print(ET.tostring(tem) ,'here')
                    set = i.find('./set')
                    get = i.find('./get')

                pass

        pat_02 = self.strip_right_left('pattern', pat)

        start = ''
        end = ''
        wo_start = False
        wo_end = False

        if (pat_02.startswith('<') or pat_02.startswith('*')) and (pat_02.endswith('>') or pat_02.endswith('*')):
            wo_start_end = True # pat_txt
            pass
        else:
            wo_start_end = False # pat_txt

        if pat_02.endswith('>') or pat_02.endswith('*'):
            wo_end = True
        else:
            wo_end = False

        if pat_02.startswith('<') or pat_02.startswith('*'):
            wo_start = True
        else:
            wo_start = False

        pat_02 = ET.XML(pat).text
        if pat_02 is not None:
            pat_02 = pat_02.strip()

        d = {
            'start': start,
            'end': end,
            'wo_start': wo_start,
            'wo_end': wo_end,
            'wo_start_end': wo_start_end,
            'pattern': pat_02,
            'template': tem,
            'index': None,
            'set_exp': set,
            'get_exp': get,
            'tem_wo_start':tem_02.startswith('<') or tem_02.startswith('*'),
            'tem_wo_end':tem_02.endswith('>') or tem_02.endswith("*")
        }

        self.mod_get_set(d)
        return d

    def strip_right_left(self, tag, pattern):
        if not isinstance(pattern, str): pattern = pattern.decode('utf-8')
        tag = tag.strip('>')
        tag = tag.strip('/')
        tag = tag.strip('<')
        pat_02 = re.sub('<'+tag+'>', '', pattern)
        pat_02 = re.sub('</'+tag+'>', '', pat_02)
        pat_02 = pat_02.strip()
        print('---', pat_02, '---')
        return pat_02

    def mod_input(self, d_list, input):
        d = d_list
        l = input.split(' ')
        if d['wo_start']:
            l = l[1:]

        if d['wo_end']:
            l = l[:-1]
            #print (l)
            #exit()

        input = ' '.join(l)

        return input

    def mod_dict_out(self, d_list, input):
        d = d_list
        l = input.split(' ')
        if d['wo_start']:
            d['start'] = l[0]
            #print(l[0])
        if d['wo_end']:
            d['end'] = l[-1]
            #print(l[-1])

        self.mod_get_set(d)

    def mod_get_set(self, d):
        set = d['set_exp']
        get = d['get_exp']
        tem = d['template']
        if set is not None:

            if d['wo_end']:
                self.memory[set.attrib['name']] = d['end']
            if d['wo_start']:
                self.memory[set.attrib['name']] = d['start']
        if get is not None:
            t = ''
            if get.attrib['name'] in self.memory:
                t = self.memory[get.attrib['name']]
            print(t,'===', ET.tostring(get))
            star = tem.find('get')
            tt = ''
            if star is not None:
                tem.remove(star)
            if d['tem_wo_end']:
                tt = tem.text.strip() + ' ' + t
            if d['tem_wo_start']:
                tt = t + ' ' + tem.text.strip()
            #d['template'] = tem.text
            n = ET.Element('template')

            n.text = tt
            tem = n
            print('>>',ET.tostring(tem), t, self.memory)
            d['template'] = tem
            #exit()


if __name__ == '__main__':

    k = Kernel()
    k.verbose(True)
    k.learn('startup.xml')
    while True:
        print(k.respond(input('> ')))