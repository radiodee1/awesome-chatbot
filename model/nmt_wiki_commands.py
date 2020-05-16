#!/usr/bin/python3

#import os
import sys
#from subprocess import Popen
#import re
#import xml.etree.ElementTree as ET
#from google import search
print(sys.path)
import importlib
#google = importlib.import_module('/usr/local/lib/python3.7/dist-packages/google/__init__.py')
try:
    from google import search
except:
    pass
#search = google.search
#from serpapi.google_search_results import GoogleSearchResults
#from googlesearch.googlesearch import GoogleSearch
#search = GoogleSearch().search
import bs4
import requests
sys.path.append('..')
#from model.settings import hparams as hp

#aiml_txt = hp['data_dir'] + '/std_startup.xml'

class Wikipedia:
    def __init__(self):
        self.topic = ''
        self.topic_old = ''
        self.text = ''

        self.url_search = 'http://find'
        self.url_stop = 'http://stop'
        self.url_start = 'https://en.wikipedia.org/wiki/'
        pass

    def search_string(self, i):
        if i.startswith(self.url_search):
            i = i[len(self.url_search):]
            self.set_topic(i)
            return i

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
            #client = GoogleSearchResults({'q':query})
            #s = client.get_dict()
            print(s)
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
    w.set_topic('xxzz')
    z = w.get_text()
    print(z)