import os
import tensorflow as tf
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import generator_utils
import sys

sys.path.append('../../..')
sys.path.append('..')
from model.settings import hparams as hp


tf.summary.FileWriterCache.clear()  # ensure filewriter cache is clear for TensorBoard events file

load_sep_files = True


@registry.register_problem
class ChatLineProblem(text_problems.Text2TextProblem):
    """Predict next line of chat from the last line. From Gutenberg texts."""

    @property
    def approx_vocab_size(self):
        return 2 ** 13  # ~8k

    @property
    def is_generate_per_split(self):
        # generate_data will NOT shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 90,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 10,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        raw_file =  hp['data_dir'] + '/raw.txt'
        file_fr = hp['data_dir'] + '/train.from'
        file_to = hp['data_dir'] + '/train.to'

        if not load_sep_files:
            if not os.path.isfile(raw_file):
                #os.system('ls -l')
                exit('cannot find file... ' + raw_file)
            with open(raw_file, 'r') as rawfp:
                prev_line = ''
                for curr_line in rawfp:
                    curr_line = curr_line.strip()
                    # poems break at empty lines, so this ensures we train only
                    # on lines of the same poem
                    if len(prev_line) > 0 and len(curr_line) > 0:
                        yield {
                            "inputs": prev_line,
                            "targets": curr_line
                        }
                    prev_line = curr_line
        else:
            if (not os.path.isfile(file_fr)) or (not os.path.isfile(file_to)):
                exit('cannot find files...' + file_fr + ' or ' + file_to)
            raw_file_fr = open(file_fr, 'r')
            raw_file_to = open(file_to, 'r')
            read_file_fr = raw_file_fr.readlines()
            read_file_to = raw_file_to.readlines()
            for i in range(len(read_file_fr)):
                prev_line = read_file_fr[i].strip()
                curr_line = read_file_to[i].strip()
                if len(prev_line) > 0 and len(curr_line) > 0:
                    yield {
                        "inputs": prev_line,
                        "targets": curr_line
                    }
            raw_file_fr.close()
            raw_file_to.close()

            # Smaller than the typical translate model, and with more regularization


@registry.register_hparams
def transformer_chat():
    #hparams = transformer.transformer_base_v2() ## comment out
    hparams = transformer.transformer_base()   ## comment in

    hparams.num_hidden_layers = 6 # 2
    hparams.hidden_size = 512 # 128
    hparams.filter_size = 2048 # 512
    hparams.num_heads = 8 #4
    hparams.attention_dropout = 0.6
    hparams.layer_prepostprocess_dropout = 0.1 #0.6
    hparams.learning_rate = 0.05

    #hparams.learning_rate_schedule = 'legacy' ## comment out

    return hparams

'''
@registry.register_hparams()
def transformer_chat_tpu():
    hparams = transformer_chat()
    transformer.update_hparams_for_tpu(hparams)
    return hparams
'''

'''
# hyperparameter tuning ranges
@registry.register_ranged_hparams
def transformer_chat_range(rhp):
    rhp.set_float("learning_rate", 0.05, 0.25, scale=rhp.LOG_SCALE)
    rhp.set_int("num_hidden_layers", 2, 4)
    rhp.set_discrete("hidden_size", [128, 256, 512])
    rhp.set_float("attention_dropout", 0.4, 0.7)
'''