#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from tensor2tensor.bin import t2t_trainer
#from tensor2tensor.bin import t2t_decoder
import sys

sys.path.append('..')
sys.path.append('../model/')

import tensorflow as tf
import argparse
from model.settings import hparams as hp
import os
import subprocess
#from tensor2tensor.serving import query as t2t_query

name = 'chat_10'

parser = argparse.ArgumentParser(
    description='Fine-tune tensor-2-tensor on your chat dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#parser.add_argument('--train', action='store_true', help='start train method.')
#parser.add_argument('--test', action='store_true', help='start test method.')
parser.add_argument('--query', action='store_true', help='start query operation.')
parser.add_argument('--export', action='store_true', help='start export operation.')
parser.add_argument('--task', help='task to start with.', default='1')
parser.add_argument('--increment', default=5000, type=int, help='default increment for trainer.')
parser.add_argument('--limit', default=10000, type=int, help='default limit for trainer.')
parser.add_argument('--no-limit', action='store_true', help='loop unconditionally through trainer.')
parser.add_argument('--name', default=name, help='run filename.')
args = parser.parse_args()

if not args.query:
    #from tensor2tensor.bin import t2t_trainer
    #from tensor2tensor.bin import t2t_decoder
    pass
else:
    from tensor2tensor.serving import query as t2t_query

train_not_test = True
task = str(int(args.task))
increment = str(int(args.increment))
limit = int(args.limit)
problem = 'chat_line_problem'
port = '8500'
p = None

counter_dir = os.path.join(hp['save_dir'], 't2t_train', args.name)
counter_path = counter_dir + '/counter'
counter = 0

checkpoint_dir = os.path.join(hp['save_dir'], 't2t_train', args.name)
checkpoint_path = checkpoint_dir + '/checkpoint'


def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

maketree(hp['data_dir'] + '/t2t_data/' + args.name + '/')
maketree(hp['data_dir'] + '/t2t_data/' + args.name + '/' + 'tmp/')

maketree(hp['save_dir'] + '/t2t_train/' + args.name + '/')
maketree(counter_dir)


export_args = [

    '--data_dir=' + hp['data_dir'] + '/t2t_data/' + args.name + '/' ,
    '--output_dir=' + hp['save_dir'] + '/t2t_train/' + args.name + '/',
    '--problem='+ problem,
    '--model=transformer' ,
    '--hparams_set=transformer_chat',

    '--t2t_usr_dir=./chat/trainer',

]

query_args = [
    sys.argv[0],
    '--port=' + port ,
    '--data_dir=' + hp['data_dir'] + '/t2t_data/' + args.name + '/',
    #'--output_dir=' + hp['save_dir'] + '/t2t_train/' + args.name + '/',
    '--problem=' + problem ,
    '--model_base_path=' + hp['save_dir'] + '/t2t_train/' + args.name +'/export/' , #'chosen/',
    '--model_name=' + name ,
    '--hparams_set=transformer_chat',
    '--server=localhost:' + port,
    '--servable_name=chat',
    '--t2t_usr_dir=./chat/trainer/',

]

server_args = [
    'tensorflow_model_server',
    '--port=' + port ,
    #'--data_dir=' + hp['data_dir'] + '/t2t_data/' ,
    #'--output_dir=' + hp['save_dir'] + '/t2t_train/' + args.name + '/',
    '--problem=' + problem ,
    '--model_base_path='  + os.getcwd() + '/' + hp['save_dir'] + '/t2t_train/' + args.name +'/export/' , #'chosen/',
    '--model_name=' + name ,
    '--hparams_set=transformer_chat',
    '--server=localhost:' + port,
    '--servable_name=chat',
    '--t2t_usr_dir=./chat/trainer/',

]

def main(argv):
    global counter
    global p

    if args.export:
        print(export_args)
        os.system('t2t-exporter ' + ' '.join(export_args))
        exit()

    print(argv, '\n','=' * 20)

    if args.query:
        try:
            tf = server_args[0]

            if os.path.isfile('/usr/bin/' + tf) or os.path.isfile('/usr/local/bin/' + tf):
                p = subprocess.Popen(server_args)
            print(argv)
            #exit()

            t2t_query.main(argv)
            exit()
        except:
            p.terminate()


if __name__ == "__main__":

    print(sys.argv)

    if args.query:
        sys.argv = query_args
        print('print:', sys.argv)
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.app.run()
        #main(sys.argv)
    elif args.export:
        #sys.argv = export_args
        print('print:', sys.argv)
        #tf.logging.set_verbosity(tf.logging.INFO)
        main(args)
        #tf.app.run()