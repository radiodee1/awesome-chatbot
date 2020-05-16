#!/usr/bin/python3

import os
import sys
import json

print(sys.path)

from googleapiclient.discovery import build

import bs4
import requests
sys.path.append('..')
#from model.settings import hparams as hp
cse_id = os.environ['CSE_ID']
api_key = os.environ['API_KEY']

def google_query(query, api_key, cse_id, **kwargs):
    query_service = build("customsearch",
                          "v1",
                          developerKey=api_key
                          )
    query_results = query_service.cse().list(q=query,    # Query
                                             cx=cse_id,  # CSE ID
                                             **kwargs
                                             ).execute()
    return query_results['items']

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
            #s = search(query, tld='com', num=20, stop=20)
            s = google_query(query, api_key=api_key, cse_id=cse_id)
            print(s)
            r = ''
            for j in s:
                j = j['link']
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
    w.set_topic('BEATLES')
    z = w.get_text()
    print(z)