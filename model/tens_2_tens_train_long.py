#!/usr/bin/python3.6

import sys
sys.path.append('..')
from tensor2tensor.bin import t2t_trainer
from model.settings import hparams
import tensorflow as tf



problem = 'chat_line_problem'
outdir = hparams['save_dir'] + '/t2t_trained_model/'
train_steps = str(75000 * 2) ## 7500

args_train = [
    '--t2t_usr_dir=' + './transformer/',
    '--problem=' + problem,
    '--data_dir=' + hparams['data_dir'] + '/t2t_data/',

    '--model=' + 'transformer',
    '--output_dir=' + outdir,
    '--hparams_set=' + 'transformer_chat',
    '--job_dir=' + outdir,
    '--train_steps=' + train_steps,

]



def main(argv):

    t2t_trainer.main(argv)

if __name__ == "__main__":

    sys.argv.extend(args_train[:])
    print(sys.argv)

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run()


