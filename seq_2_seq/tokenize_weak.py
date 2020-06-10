#!/usr/bin/env python3

import sys
sys.path.append('..')
import re
import unidecode
from settings import hparams

timeframes = ['input']

to_lower = True
test_on_screen = False
remove_caps = True

def format(content, do_tokenize=False):
    c = content.lower().strip()
    c = unidecode.unidecode(c)
    c = re.sub('[][)(\n\r#@*^><:|{}]',' ', c)
    c = re.sub("[\"`]","'",c)

    c = c.split(' ')

    cy = []
    for z in c:
        begin = re.findall(r"^[']+([^']*)", z)
        end = re.findall(r"(\w+)[']+$", z)
        w_period = re.findall(r"^(\w+)'\.$", z)
        b_e_w_period = re.findall(r"^'(\w+)'\.$", z)
        both = re.findall(r"^[']+([^']*)[']+$",z)
        amp = re.findall(r"&(\w+);",z) ## anywhere in word
        link = re.findall(r"^http(\w+)",z)
        link2 = re.findall(r"^\(http(\w+)",z)
        www = re.findall(r"^www",z)
        odd = re.findall(r"([$%0123456789+=^;:~_/\\])(\w*)", z)
        double = re.findall(r"(['])(['])+", z)

        #if test_on_screen: print(z,begin,end, both)


        if len(odd) > 0:
            ## this sometimes consumes punctuation
            pass
        elif len(double) > 0 and len(w_period) == 0 and (len(begin) == 0 or len(end) == 0):
            l = z.split("'")
            if len(l) > 2:
                z = re.sub('[\']','',z)
                #print(z)
                cy.append(z)
            else:
                cy.append(z)
        elif len(both) > 0:
            cy.append("'")
            cy.append(both[0])
            cy.append("'")
        elif len(b_e_w_period) > 0:
            cy.append("'")
            cy.append(b_e_w_period[0])
            cy.append("'")
            cy.append(".")
        elif len(w_period) > 0:
            #print(w_period)
            cy.append(w_period[0])
            cy.append("'")
            cy.append(".")
        elif len(begin) > 0:
            cy.append("'")
            cy.append(begin[0])
        elif len(end) > 0:
            cy.append(end[0])
            cy.append("'")
        elif len(amp) > 0 or len(link) > 0 or len(link2) > 0 or len(www) > 0 or z == 'newlinechar':
            # do not append z!!
            pass

        else:
            pass
            cy.append(z)


    x = ' '.join(cy)

    x = re.sub('[!]', ' ! ', x)
    x = re.sub('[?]', ' ? ', x)
    x = re.sub('[,]', ' , ', x)
    x = re.sub('[-]', ' ', x)
    x = re.sub('[/]', '', x)
    x = re.sub("[`]", "'", x)
    x = re.sub('[.]', ' . ', x)
    if hparams['embed_size'] is not None and hparams['embed_size'] != 0:
        x = re.sub("[']", " ' ", x) #contractions!!

    c = x.split()

    cx = []
    for i in range(len(c)):
        cc = c[i].strip()

        if i < len(c) - 1 and cc != c[i + 1].lower():
            ## skip elipses and repeats.
            cx.append(cc)
        elif i + 1 == len(c):
            cx.append(cc)

    x = ' '.join(cx)


    #if test_on_screen: print(x)
    return x

if __name__ == '__main__':
    ## try one line of text
    test_on_screen = True
    print(format('here There we are www.here.com ... ? ! ? ?'))
    print(format("it's \"a\" very ''very' 'bad 'thing'."))
    print(format(' something like %%%, or $$$ , right?'))
    print(format("it's like 1 or ' me ' or 23ish. $omething "))
    print(format("."))