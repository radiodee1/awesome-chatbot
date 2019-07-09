#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.bin import t2t_decoder
import sys

#sys.path.append('../model/')
sys.path.append('..')

import tensorflow as tf
import argparse
from model.settings import hparams as hp
import os

parser = argparse.ArgumentParser(
    description='Fine-tune tensor-2-tensor on your babi dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--train', action='store_true', help='start train method.')
parser.add_argument('--test', action='store_true', help='start test method.')
parser.add_argument('--task', help='task to start with.', default='1')
parser.add_argument('--prep', action='store_true', help='do data prep step.')
args = parser.parse_args()

train_not_test = True
task = str(int(args.task))

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

maketree(hp['data_dir'] + '/t2t_data/')
maketree(hp['data_dir'] + '/t2t_data/tmp/')

maketree(hp['save_dir'] + '/t2t_train/babi/')

trainer_args = [
    sys.argv[0],
    '--generate_data' ,
    '--data_dir=' + hp['data_dir'] + '/t2t_data/',
    '--tmp_dir=' + hp['data_dir'] + '/t2t_data/tmp/',
    '--output_dir=' + hp['save_dir'] + '/t2t_train/babi/',
    '--problem=babi_qa_concat_task' + task + '_10k' ,
    '--model=transformer',
    '--hparams_set=transformer_base',
    #'--train_steps=1000',
    #'--eval_steps=500',
    #'trainer args'
]

decoder_args = [
    sys.argv[0],
    #'--generate_data' ,
    '--data_dir=' + hp['data_dir'] + '/t2t_data/' ,
    '--tmp_dir=' + hp['data_dir'] + '/t2t_data/tmp/',
    '--output_dir=' + hp['save_dir'] + '/t2t_train/babi/',
    '--problem=babi_qa_concat_task' + task + '_10k' ,
    '--model=transformer' ,
    '--hparams_set=transformer_base',
    #'--train_steps=1000',
    '--eval_steps=500',
    #'decoder args'
]

prepare_args = [
    sys.argv[0],

]

def main(argv):
    print(argv)
    if train_not_test:
        t2t_trainer.main(argv)
    else:
        t2t_decoder.main(argv)
    pass

if __name__ == "__main__":
    print(sys.argv)
    if args.train:
        train_not_test = True
        sys.argv = trainer_args
    elif args.test:
        train_not_test = False
        sys.argv = decoder_args
    print('print:', sys.argv)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()