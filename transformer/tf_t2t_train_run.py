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
#from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import

#from tensor2tensor.serving import make_request_fn, validate_flags

from tensor2tensor.serving import serving_utils
from tensor2tensor.utils import hparam
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

parser = argparse.ArgumentParser(
    description='Run NMT for chat program.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', default='chat_01', help='run filename.')  # default = 'babi' <-- ??
args = parser.parse_args()

print('args:',args)

train_not_test = True
#task = str(int(args.task))
#increment = str(int(args.increment))
#limit = int(args.limit)
problem = 'chat_line_problem'
port = '9002'
p = None
request_fn = None
problem_hp = None

counter_dir = os.path.join(hp['save_dir'], 't2t_train', args.name)
counter_path = counter_dir + '/counter'
counter = 0

checkpoint_dir = os.path.join(hp['save_dir'], 't2t_train', args.name)
checkpoint_path = checkpoint_dir + '/checkpoint'

flags.DEFINE_boolean('cloud_mlengine_model_name', False, 'skip ml engine!')
flags.DEFINE_string('server', 'localhost' , 'server location.')
flags.DEFINE_string('servable_name', 'chat' , 'servable name.')
flags.DEFINE_string('t2t_usr_dir', './chat/trainer/' , 'usr dir name.')
flags.DEFINE_string('problem', problem , 'problem name.')
flags.DEFINE_string('data_dir', hp['data_dir'] + '/t2t_data/' + args.name + '/' , 'data dir name.')
flags.DEFINE_string('model_base_path', hp['save_dir'] + '/t2t_train/' + args.name + '/export/' , 'data dir name.')
flags.DEFINE_string('model_name', 'chat' , 'model name.')
flags.DEFINE_string('port',  port, 'server location.')

flags.DEFINE_integer('timeout_secs', 100, 'timeout secs.')
flags.DEFINE_string('inputs_once', None , 'input.')


server_args = [
    'tensorflow_model_server',
    '--port=' + port,
    #'--data_dir=' + hp['data_dir'] + '/t2t_data/' ,
    #'--output_dir=' + hp['save_dir'] + '/t2t_train/' + args.name + '/',
    '--problem=' + problem,
    '--model_base_path=' + os.getcwd() + '/' + hp['save_dir'] + '/t2t_train/' + args.name + '/export/',  # 'chosen/',
    '--model_name=' + 'chat',
    '--hparams_set=transformer_chat',
    '--server=localhost:' + port,
    '--servable_name=chat',
    '--t2t_usr_dir=./chat/trainer/',

]

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def validate_flags():
    """Validates flags are set to acceptable values."""
    if FLAGS.cloud_mlengine_model_name:
        assert not FLAGS.server
        assert not FLAGS.servable_name
    else:
        assert FLAGS.server
        assert FLAGS.servable_name


def make_request_fn():
    """Returns a request function."""
    if FLAGS.cloud_mlengine_model_name:
        request_fn = serving_utils.make_cloud_mlengine_request_fn(
            credentials=GoogleCredentials.get_application_default(),
            model_name=FLAGS.cloud_mlengine_model_name,
            version=FLAGS.cloud_mlengine_model_version)
    else:

        request_fn = serving_utils.make_grpc_request_fn(
            servable_name=FLAGS.servable_name,
            server=FLAGS.server,
            timeout_secs=FLAGS.timeout_secs)
    return request_fn


def main_setup():
    #try:
    global request_fn
    global problem_hp
    #p = subprocess.Popen(server_args,shell=True)

    #tf.logging.set_verbosity(tf.logging.INFO)
    validate_flags()
    usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
    problem_hp = registry.problem(FLAGS.problem)
    #print(FLAGS.data_dir, '<---')
    hparams = hparam.HParams(
        data_dir=os.path.expanduser(FLAGS.data_dir))
    problem_hp.get_hparams(hparams)
    request_fn = make_request_fn()

    #except KeyboardInterrupt:
    #    pass
    #finally:
        #p.terminate()
    #    pass

def predict_once(inputs):
    global request_fn
    global problem_hp
    outputs = serving_utils.predict([inputs], problem_hp, request_fn)
    outputs, = outputs
    output, score = outputs

    return output


class NMT:
    def __init__(self):
        self.name = ''
        self.p = None

    def setup_for_interactive(self):
        try:
            print(server_args, '<---')
            self.p = subprocess.Popen(server_args, shell=True)
            print(self.p)

            main_setup()
            #tf.app.run(main_setup)
            #setup()
            # main(sys.argv)
        except KeyboardInterrupt:
            pass
        finally:
            self.p.terminate()

    def get_sentence(self, i):
        return predict_once(i)

    def loop(self):
        while True:
            try:
                i = input("> ")
                self.get_sentence(i)
            except EOFError:
                print()
                exit()
            except KeyboardInterrupt:
                print()
                exit()

if __name__ == '__main__':
    g = NMT()
    g.setup_for_interactive()
    g.loop()
