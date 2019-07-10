#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.bin import t2t_decoder
import sys

sys.path.append('..')

import tensorflow as tf
import argparse
from model.settings import hparams as hp
import os
#import json

parser = argparse.ArgumentParser(
    description='Fine-tune tensor-2-tensor on your babi dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--train', action='store_true', help='start train method.')
parser.add_argument('--test', action='store_true', help='start test method.')
parser.add_argument('--task', help='task to start with.', default='1')
parser.add_argument('--increment', default=50, type=int, help='default increment for trainer.')
parser.add_argument('--limit', default=10000, type=int, help='default limit for trainer.')
args = parser.parse_args()

train_not_test = True
task = str(int(args.task))
increment = str(int(args.increment))
limit = int(args.limit)

counter_dir = os.path.join(hp['save_dir'], 't2t_train', 'babi')
counter_path = counter_dir + '/counter'
counter = 0

checkpoint_dir = os.path.join(hp['save_dir'], 't2t_train', 'babi')
checkpoint_path = checkpoint_dir + '/checkpoint'


def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

maketree(hp['data_dir'] + '/t2t_data/')
maketree(hp['data_dir'] + '/t2t_data/tmp/')

maketree(hp['save_dir'] + '/t2t_train/babi/')
maketree(counter_dir)

trainer_args = [
    sys.argv[0],
    '--generate_data' ,
    '--data_dir=' + hp['data_dir'] + '/t2t_data/',
    '--tmp_dir=' + hp['data_dir'] + '/t2t_data/tmp/',
    '--output_dir=' + hp['save_dir'] + '/t2t_train/babi/',
    '--problem=babi_qa_concat_task' + task + '_10k' ,
    '--model=transformer',
    '--hparams_set=transformer_base',
    '--eval_steps=5',
]

train_steps_arg = '--train_steps='

decoder_args = [
    sys.argv[0],
    #'--generate_data' ,
    '--data_dir=' + hp['data_dir'] + '/t2t_data/' ,
    '--tmp_dir=' + hp['data_dir'] + '/t2t_data/tmp/',
    '--output_dir=' + hp['save_dir'] + '/t2t_train/babi/',
    '--problem=babi_qa_concat_task' + task + '_10k' ,
    '--model=transformer' ,
    '--hparams_set=transformer_base',

    '--eval_steps=500',
    '--decode_to_file=' + hp['save_dir'] + '/t2t_train/' + 'decode_file.txt',
    '--score_file=' + hp['data_dir'] + '/t2t_data/' + 'test_tab.txt',

]



def main(argv):
    global counter

    print(argv, '\n','=' * 20)
    if train_not_test:
        while counter < limit:

            tf.flags.FLAGS.set_default('train_steps', counter + args.increment)
            tf.flags.FLAGS.train_steps = counter + args.increment
            print('flag:', tf.flags.FLAGS.get_flag_value('train_steps', 5), str(counter + args.increment))

            t2t_trainer.main(argv)

            counter += args.increment
            print('=' * 50, counter, limit, '=' * 50)

    else:
        t2t_decoder.main(argv)
    pass

if __name__ == "__main__":

    print(sys.argv)
    if args.train:

        if os.path.isfile(checkpoint_path):
            try:
                with open(checkpoint_path,'r') as z:
                    z = z.readline()
                    print(z)
                    z = z.split('\n')[0]
                    z = z.strip()
                    z = z.split(':')
                    z = z[-1]
                    z = z.strip()
                    z = z.strip('"')
                    z = z.split('-')
                    z = z[-1]
                    z = int(z)
                    print(z)
                    counter = z
                    pass
            except :
                print('could not load counter from checkpoint.')
                pass
        train_not_test = True
        sys.argv = trainer_args + [train_steps_arg + str(counter + args.increment)]

        tf.logging.set_verbosity(tf.logging.INFO)

        tf.app.run()

    elif args.test:
        train_not_test = False
        sys.argv = decoder_args
        print('print:', sys.argv)
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.app.run()

