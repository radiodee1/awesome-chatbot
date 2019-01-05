#!/usr/bin/python3.6

import os
import sys


'''

cp data/vocab.big.txt saved/saved_vocab.big.txt
cp data/vocab.babi.txt saved/saved_vocab.babi.txt
#cp data/vocab.to saved/saved_vocab.to
cp model/settings.py saved/saved_settings.py.txt
cp data/embed.txt saved/saved_embed.txt

#cd saved
zip -r vocab.zip saved/saved_*.txt
rm saved/saved_*.txt

mv vocab.zip ../
'''
print(sys.argv)

if len(sys.argv) == 1:
    print('specify "basename" as commandline argument.')
    exit()

path = sys.argv[1]
print(path, len(sys.argv[1:]))

basename = path.strip().split('/')[-1]
basename = basename.strip().split('.')[0]

print(basename)
if len(basename) < 1:
    print('specify "basename" as commandline argument.')
    exit()

## redundant add to zip.
for i in sys.argv[1:]:
    os.system('zip -r ' + basename + '.zip '+ i)


os.system('zip -r ' + basename + '.zip ' + './saved/' + basename + '*' )
os.system('zip -r ' + basename + '.zip ' + './data/vocab.*.txt ./data/embed.txt ./model/settings.py')
os.system('mv ' + basename + '.zip ../.')
