#!/usr/bin/python3.6

import sys
sys.path.append('..')
#from tensor2tensor.bin import t2t_decoder
from tensor2tensor.serving import query
from model.settings import hparams
import tensorflow as tf
import os
import subprocess

pwd = os.getcwd()
#folder = "1553282176"
problem = 'chat_line_problem'
outdir = hparams['save_dir'] + '/t2t_trained_model/export/'

args_query = [
    '--t2t_usr_dir=' + './transformer/',
    '--problem=' + problem,
    '--server=localhost:9000',
    '--servable_name=' + problem,
    '--data_dir=' + hparams['data_dir'] + '/t2t_data/',

]
args_server = [
    'tensorflow_model_server',
    '--port=9000',
    '--model=' + 'transformer',
    '--model_base_path=' + 'file://' + pwd + '/' + outdir,
    '--model_name=' + problem , #'transformer_chat',
    '--t2t_usr_dir=' + './transformer/',
    '--problem=' + problem,

]



class NMT:
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.ERROR)

        sys.argv.extend(args_query)

        z = tf.app.run(main=self.main)


    def main(self, argv):
        subprocess.Popen(args_server)

        query.main(argv)

    def setup_for_interactive(self):
        #sys.stdin = ["hello there"]
        #tf.app.run(main=self.main)
        pass

    def get_sentence(self, inval):

        pass

if __name__ == "__main__":
    nmt = NMT()
    nmt.setup_for_interactive()



