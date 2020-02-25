#!/usr/bin/python3

import os
import sys
from subprocess import Popen
import re
import xml.etree.ElementTree as ET
from googlesearch import search
import bs4
import requests
sys.path.append('..')
from model.settings import hparams as hp

aiml_txt = hp['data_dir'] + '/std_startup.xml'

class Wikipedia:
    def __init__(self):
        self.topic = ''
        self.topic_old = ''
        self.text = ''

        self.url_start = 'https://en.wikipedia.org/wiki/'
        pass

    def print_aiml_name(self):
        print(aiml_txt)
        tree = ET.parse(aiml_txt)
        root = tree.getroot()
        print(root)

    def set_topic(self, topic):
        self.topic_old = self.topic
        self.topic = topic
        pass

    def get_text(self):
        if self.topic == self.topic_old:
            return self.text
        else:
            ## search for text ##
            query = self.topic
            s = search(query, tld='com', num=20, stop=20)
            r = ''
            for j in s:
                print(j)
                if j.startswith(self.url_start):
                    rr = requests.get(j).text
                    ## just paragraph tags!! ##
                    wiki = bs4.BeautifulSoup(rr, 'html.parser')
                    for i in wiki.select('p'):
                        if '<' not in i.getText() and '>' not in i.getText():
                            r += i.getText() + ' '
                    break
            self.text = r.strip()
            print(len(self.text.split(' ')), 'num tokens')

            return self.text


if __name__ == '__main__':
    w = Wikipedia()
    w.print_aiml_name()
    w.set_topic('xxzz')
    z = w.get_text()
    print(z)