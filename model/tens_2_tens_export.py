#!/usr/bin/python3.6

import sys
sys.path.append('..')
from tensor2tensor.serving import export
from model.settings import hparams
import tensorflow as tf



problem = 'chat_line_problem'
outdir = hparams['save_dir'] + '/t2t_trained_model/'
#train_steps = str(7500) ## 7500

args_train = [
    '--t2t_usr_dir=' + './transformer/',
    '--problem=' + problem,
    '--data_dir=' + hparams['data_dir'] + '/t2t_data/',

    '--model=' + 'transformer',
    '--output_dir=' + outdir ,
    '--hparams_set=' + 'transformer_chat',
    '--decode_hparams=' + 'beam_size=4,alpha=0.6',
    '--model_name=' + 'transformer_chat',
    #'--train_steps=' + train_steps,

]



def main(argv):

    export.main(argv)

if __name__ == "__main__":

    sys.argv.extend(args_train[:])
    print(sys.argv)

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run()


