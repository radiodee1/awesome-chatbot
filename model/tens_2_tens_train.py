#!/usr/bin/python3.6

import sys
sys.path.append('..')
from tensor2tensor.bin import t2t_trainer
from model.settings import hparams
import tensorflow as tf
import os


problem = 'chat_line_problem'
outdir = hparams['save_dir'] + '/t2t_trained_model/'

args_train = [
    '--t2t_usr_dir=' + '../transformer/',
    '--problem=' + problem,
    #'--data_dir=' + outdir + '/data/', # hparams['save_dir'] + '/t2t_data/',
    #'--tmp_dir=' + hparams['data_dir'] +'/tmp/' #,
    '--model=' + 'transformer',
    '--output_dir=' + outdir, # hparams['save_dir'] + '/t2t_saved/'
    '--hparams_set=' + 'transformer_chat',
    '--job_dir=' + outdir, # hparams['save_dir'] + '/t2t_saved/',
    '--train_steps='+ '10',
    '--generate-data'
]



def main(argv):

    t2t_trainer.main(argv)

if __name__ == "__main__":

    sys.argv.extend(args_train[:])
    print(sys.argv)

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run()


