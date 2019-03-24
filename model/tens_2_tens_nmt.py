#!/usr/bin/python3.6

import sys
sys.path.append('..')
#from tensor2tensor.bin import t2t_decoder
import model.tens_2_tens_query as query
from model.settings import hparams
import tensorflow as tf
import os
import subprocess
import json
import requests


pwd = os.getcwd()
#folder = "1553282176"
problem = 'chat_line_problem'
outdir = hparams['save_dir'] + '/t2t_trained_model/export/'
url = "http://127.0.0.1:9001/v1/models/" + problem + ':predict'
headers = {"content-type": "application/json"}

args_query = [
    '--t2t_usr_dir=' + './transformer/',
    '--problem=' + problem,
    '--server=localhost:9000',
    '--servable_name=' + problem,
    '--data_dir=' + hparams['data_dir'] + '/t2t_data/',
    #'--inputs_once=' + '"true"',

]

args_server = [
    'tensorflow_model_server',
    '--port=9000',
    '--rest_api_port=9001',
    '--model=' + 'transformer',
    '--model_base_path=' + 'file://' + pwd + '/' + outdir,
    '--model_name=' + problem , #'transformer_chat',
    '--t2t_usr_dir=' + './transformer/',
    '--problem=' + problem,

]



class NMT:
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.INFO)

        sys.argv.extend(args_query)

        #tf.app.run(main=self.main,argv=sys.argv)
        self.args = []
        #self.main(sys.argv)
        #self.q = None
        self.main(sys.argv)


    def main(self, argv):
        self.args = argv
        self.args.extend(args_query)
        subprocess.Popen(args_server)

        pass

    def setup_for_interactive(self):
        #sys.stdin = ["hello there"]
        #tf.app.run(main=self.main)

        pass

    def get_sentence(self, inval):

        #argv.extend(args_query)

        #print(self.args, 'here 1')
        inval = inval.replace('"', '\"')
        args_local = self.args + ['--inputs_once="' + inval + '"']

        print(args_local)

        z = query.main(args_local)
        return z


if __name__ == "__main__":
    nmt = NMT()
    nmt.setup_for_interactive()
    while True:
        inval = input('>==>')
        outval = nmt.get_sentence(inval)

        print(outval,'===')



