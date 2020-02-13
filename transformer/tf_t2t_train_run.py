#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from tensor2tensor.bin import t2t_trainer
# from tensor2tensor.bin import t2t_decoder
import sys

sys.path.append('..')
sys.path.append('../model/')

#import tensorflow as tf
import argparse
from model.settings import hparams as hp
import os
import subprocess
import tensorflow as tf
import datetime
#from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import

#from tensor2tensor.serving import make_request_fn, validate_flags

from tensor2tensor.serving import serving_utils
from tensor2tensor.utils import hparam
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir

flags = tf.flags
FLAGS = flags.FLAGS

name = 'chat_10'
include_date = False
include_questionmark = False

parser = argparse.ArgumentParser(
    description='Run NMT for chat program.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', default=name, help='run filename.')  # default = 'chat_10' <-- ??
args = parser.parse_args()

print('args:',args)

train_not_test = True
#task = str(int(args.task))
#increment = str(int(args.increment))
#limit = int(args.limit)
problem = 'chat_line_problem'
port = '8500'
port_rest = '8501'
p = None
request_fn = None
problem_hp = None

counter_dir = os.path.join(hp['save_dir'], 't2t_train', args.name)
counter_path = counter_dir + '/counter'
counter = 0

checkpoint_dir = os.path.join(hp['save_dir'], 't2t_train', args.name)
checkpoint_path = checkpoint_dir + '/checkpoint'

server = 'localhost:'
servable_name = args.name # 'chat_10'
data_dir = os.getcwd() + '/' + hp['data_dir'] + 't2t_data/' + args.name + '/'
t2t_usr_dir = './chat/trainer/'

flags.DEFINE_boolean('cloud_mlengine_model_name', False, 'skip ml engine!')
flags.DEFINE_string('server', 'localhost' , 'server location.')
flags.DEFINE_string('servable_name', servable_name , 'servable name.')
flags.DEFINE_string('t2t_usr_dir', t2t_usr_dir, 'usr dir name.')
flags.DEFINE_string('problem', problem , 'problem name.')
flags.DEFINE_string('data_dir', data_dir , 'data dir name.')
flags.DEFINE_string('model_base_path', hp['save_dir'] + 't2t_train/' + args.name + '/export/' , 'data dir name.')
flags.DEFINE_string('model_name', args.name , 'model name.')
flags.DEFINE_integer('port', int( port), 'server location.')
#flags.DEFINE_integer('rest_api_port', int( port_rest), 'server location.')
flags.DEFINE_integer('timeout_secs', 100, 'timeout secs.')
flags.DEFINE_string('inputs_once',None , 'input.')

#FLAGS = flags.FLAGS

server_args = [
    'tensorflow_model_server',
    '--port=' + port,
    #'--rest_api_port=' + port_rest,
    #'--data_dir=' + hp['data_dir'] + '/t2t_data/' ,
    #'--output_dir=' + hp['save_dir'] + '/t2t_train/' + args.name + '/',
    #'--problem=' + problem,
    #'--path=' + os.getcwd() + '/' + hp['save_dir'] + '/t2t_train/' + args.name + '/export/',
    '--model_base_path=' + os.getcwd() + '/' + hp['save_dir'] + 't2t_train/' + args.name + '/export/',  # 'chosen/',
    '--model_name=' + args.name ,
    #'--hparams_set=transformer_chat',
    '--server=localhost:' + port ,
    #'--servable_name=chat',
    #'--t2t_usr_dir=./chat/trainer/',

]

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

def validate_flags():
    assert server
    assert servable_name

def make_request_fn():
    """Returns a request function."""
    if True:
        request_fn = serving_utils.make_grpc_request_fn(
            servable_name=servable_name,
            server=server + port,
            timeout_secs=100)

    #print(request_fn,'<-6')
    return request_fn


def main_setup():
    #try:
    global request_fn
    global problem_hp

    usr_dir.import_usr_dir(t2t_usr_dir)
    problem_hp = registry.problem(problem)
    hparams = hparam.HParams(
        data_dir=os.path.expanduser(data_dir))
    problem_hp.get_hparams(hparams)
    request_fn = make_request_fn()



def predict_once(inputs):
    global request_fn
    global problem_hp
    #print(problem_hp, request_fn, 'request')
    outputs = serving_utils.predict([inputs], problem_hp, request_fn)
    outputs, = outputs
    output, score = outputs

    return output


class NMT:
    def __init__(self):
        self.name = ''
        self.p = None
        #print('start <')

    def setup_for_interactive(self):
        #print(server_args, '<---')
        tff = server_args[0]

        if os.path.isfile('/usr/bin/' + tff) or os.path.isfile('/usr/local/bin/' + tff):
            self.p = subprocess.Popen(server_args, shell=False)
        #print(self.p)
        tf.logging.set_verbosity(tf.logging.INFO)

        main_setup()

    def get_sentence(self, i):
        try:
            z = predict_once(i)
            if '.' in z:
                z = z.split('.')[0] ## take first sentence
            z = z.strip()
            if z.endswith('eol'):
                z = z[:-len('eol')]
            return z
        except (EOFError, KeyboardInterrupt):
            self.p.terminate()
            print()
            exit()
        except NotImplementedError as xx:
            pass
            print('terminate', type(xx).__name__)
            self.p.terminate()


    def loop(self):

        while True:
            if include_date:
                now = datetime.datetime.now()
                time = now.strftime("%I:%M %p")
                date = now.strftime("%B %d, %Y")

            try:
                i = input("> ")
                if include_questionmark:
                    if not i.endswith('?'):
                        i = i + '?'
                if include_date:
                    i = "It's " + date + ' ' + time + '. ' + i
                print(i)
                z = self.get_sentence(i.lower())
                print(z)
            except (EOFError, KeyboardInterrupt):
                self.p.terminate()
                print()
                exit()
            except:
                self.p.terminate()

if __name__ == '__main__':

    try:
        g = NMT()
        g.setup_for_interactive()
        g.loop()
    except:
        pass
