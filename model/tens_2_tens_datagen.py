#!/usr/bin/python3.6

import sys
sys.path.append('..')
from tensor2tensor.bin import t2t_datagen
from model.settings import hparams
import tensorflow as tf
import os


problem = 'chat_line_problem'

args_datagen = [
    '--t2t_usr_dir=' + './transformer/',
    '--problem=' + problem,
    '--data_dir=' + hparams['data_dir'] + '/t2t_data/',
    '--tmp_dir=' + hparams['data_dir'] +'/tmp/' #,
    #'--helpfull'
]

def main(argv):
    #argv.extend(args_datagen)

    t2t_datagen.main(argv)



if __name__ == "__main__":

    sys.argv.extend(args_datagen)
    print(sys.argv)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()


