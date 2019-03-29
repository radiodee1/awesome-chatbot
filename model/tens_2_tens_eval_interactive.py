#!/usr/bin/python3.6

import sys
sys.path.append('..')
from tensor2tensor.bin import t2t_decoder
from model.settings import hparams
import tensorflow as tf
import os
import argparse


problem = 'chat_line_problem'
outdir = hparams['save_dir'] + '/t2t_trained_model/'

args_decode = [
    '--t2t_usr_dir=' + './transformer/',
    '--problem=' + problem,
    '--data_dir=' + hparams['data_dir'] + '/t2t_data/',

    '--model=' + 'transformer',
    '--output_dir=' + outdir,
    '--hparams_set=' + 'transformer_chat',
    #'--job_dir=' + outdir,
    #'--decode_hparams=' + 'beam_size=14,alpha=0.6',
    '--decode_interactive' ,
    #'--hparams=' + 'num_hidden_layers=4,hidden_size=512' ## hparams are not tunable
]

beam_size = 4
alpha = 0.6


def main(argv):


    #t2t_decoder.main(argv)
    t2t_decoder.main(args_decode)

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Eval options...')
    parser.add_argument('--beam', help='enter beam width integer.')
    parser.add_argument('--alpha', help='float for alpha value.')
    args = parser.parse_args()
    args = vars(args)
    print(args)

    if args['beam'] is not None:
        beam_size = int(args['beam'])
        args.pop('beam')

    if args['alpha'] is not None:
        alpha = float(args['alpha'])
        args.pop('alpha')

    last_arg = ('--decode_hparams=' +
                'beam_size=' + str(beam_size) +
                ',alpha=' + str(alpha) # +
                #',decode_length=5'
                )
    args_decode.append(last_arg)

    sys.argv.extend(args_decode)
    #print(sys.argv)

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run()


