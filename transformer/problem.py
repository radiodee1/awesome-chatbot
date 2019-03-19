#!/usr/bin/python3.6

import sys

sys.path.append('..')

from model.settings import hparams
import tensorflow as tf

DATAFILE = hparams['data_dir'] + '/t2t_data/' + 'raw.txt'

import os
import tensorflow as tf
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import generator_utils

tf.summary.FileWriterCache.clear()  # ensure filewriter cache is clear for TensorBoard events file


@registry.register_problem
class ChatLineProblem(text_problems.Text2TextProblem):
    """Predict next line of poetry from the last line. From Gutenberg texts."""

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
        with open(DATAFILE, 'r') as rawfp:

            for curr in rawfp:
                raw = curr.split('\t')
                if len(raw) > 1:
                    prev_line = raw[0]
                    curr_line = raw[1]
                    prev_line = prev_line.strip()
                    curr_line = curr_line.strip()
                    # poems break at empty lines, so this ensures we train only
                    # on lines of the same poem
                    if len(prev_line) > 0 and len(curr_line) > 0:
                        yield {
                            "inputs": prev_line,
                            "targets": curr_line
                        }


# Smaller than the typical translate model, and with more regularization
@registry.register_hparams
def transformer_chat():
    hparams = transformer.transformer_base()
    hparams.num_hidden_layers = 2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.num_heads = 4
    hparams.attention_dropout = 0.6
    hparams.layer_prepostprocess_dropout = 0.6
    hparams.learning_rate = 0.05
    return hparams


@registry.register_hparams
def transformer_chat_tpu():
    hparams = transformer_chat()
    transformer.update_hparams_for_tpu(hparams)
    return hparams


# hyperparameter tuning ranges
@registry.register_ranged_hparams
def transformer_chat_range(rhp):
    rhp.set_float("learning_rate", 0.05, 0.25, scale=rhp.LOG_SCALE)
    rhp.set_int("num_hidden_layers", 2, 4)
    rhp.set_discrete("hidden_size", [128, 256, 512])
    rhp.set_float("attention_dropout", 0.4, 0.7)
