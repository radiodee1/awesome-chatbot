# Statistics and File Naming:

This page is for results of the various tests in the BABI set and for explaining the naming conventions that we're trying to stick to. The  table may be incomplete.

#### Babi:
There are 20 tests. One at a time we will try to fill in the results that we achieve in the table below. Also below is a list of filenames. Though one model is meant to complete all the tests with the same saved weights, we will assume that each test is done separately. At the end we may include a result for testing on all the categories at once.

The babi data set is downloaded in the form of 20 categories. There is the basic babi english data set and the 10k data set. We will use the 10k data set. Each category has 10,000 training examples and 1000 testing examples. For our purposes we have divided up each category into 10,000 training examples along with 500 validation set examples and 500 testing set examples. Scores will be filed in pairs. The first score in the pair is the validation set score and the second is the test set score.

* `baseline_dmn` This would be a figure for comparison that was taken from literature on the subject. It does not reflect work done with this project. It is generally used with heavy supervision. [See here](https://arxiv.org/pdf/1506.07285.pdf) and [here.](https://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/#initial-experiments)
* `small` Here the embedding weights only contain words that are in the babi corpus. The embeddings will not be trainable. The word vectors would come from the glove download. There are 158 words used in the babi corpus, and with three special tokens ('sol', 'eol', and 'unk') there are 161 vocabulary words. This would be the classic configuration for the babi test.
* `small_tr` This would be the same small set of embeddings, but not using embedding vectors from the glove download. Embeddings are not pre-trained and are trained by the pytorch model.
* `lrg` This column reflects testing done with 100 hidden units and a large set of non trainable embeddings. The individual word vectors will come from the glove download. The words in the vocabulary would include those not specific to the question answering task.

The table is included here:

 |   | baseline_dmn | small | small_tr | lrg |
|-|-|-|-|-|
 | QA1: Single Supporting Fact | 100 | 100/100 | 100/100 | 100/100 |
 | QA2: Two Supporting Facts | 98.2 | 0 | 0 | 0 |
 | QA3: Three Supporting Facts | 95.2 | 0 | 0 | 0 |
 | QA4: Two Argument Relations | 100 | 0 | 100/100 | 0 |
 | QA5: Three Argument Relations | 99.3 | 0 | 0 | 0 |
 | QA6: Yes/No Questions | 100 | 0 | 99.8/100 | 0 |
 | QA7: Counting | 96.9 | 0 | 0 | 0 |
 | QA8: Lists/Sets | 96.5 | 0 | 0 | 0 |
 | QA9: Simple Negation | 100 | 98.20/97.20* | 100/99.40 | 0 |
 | QA10: Indefinite Knowledge | 97.5 | 0 | 0 | 0 |
 | QA11: Basic Coreference | 99.9 | 0 | 0 | 0 |
 | QA12: Conjunction | 100 | 100/100 | 100/100 | 0 |
 | QA13: Compound Coreference | 99.8 | 0 | 0 | 0 |
 | QA14: Time Reasoning | 100 | 0 | 0 | 0 |
 | QA15: Basic Deduction | 100 | 0 | 0 | 0 |
 | QA16: Basic Induction | 99.4 | 0 | 0 | 0 |
 | QA17: Positional Reasoning | 59.6 | 0 | 0 | 0 |
 | QA18: Size Reasoning | 95.3 | 0 | 0 | 0 |
 | QA19: Path Finding | 34.5 | 0 | 0 | 0|
 | QA20: Agent's Motivation | 100 | 0 | 0 | 0 |
 | Average | 93.605 | 0 | 0 | 0 |
 | Combined | 0 | 0 | 0 | 0 |

_*_ -- these results may need to be revisited.
#### Training and Testing:
All results at this time benefit from weak or no supervision during training.

#### The `model` Directory:

This is a list of the models from the project. The first few are seq2seq models. In the case where the model is for seq2seq the two languages are both english. The reason for this is that the original project was aimed at making a neural network chatbot.
It was felt that a seq2seq translation model might function as a chatbot if both languages were english.
* `model/model.py` This is a Keras seq2seq model that uses a simple attention mechanism. The attention code is found in the file `model/attention_decoder.py`.
* `model/pytorch.py` This is the first pytorch model. It too is a seq2seq model. 
* `model/babi.py` This is an early version of the pytorch code for training on the babi data set. It does not have all the bells and whistles that are present in the later babi pytorch model.
* `model/babi_ii.py` This is the current version of the code meant to work with the babi dataset.
* `model/settings.py` This file holds all the hyper parameters for all the models from various versions of the project.
* `model/tokenize_weak.py` This file holds some common code for tokenizing sentences. It is used by all the models.

#### Utility Scripts in `model`:
* `model/do_make_train_test_from_babi.py` This must be run before executing any training on the babi set. This script also allows you to select which babi challenge you want to work on.
* `model/do_make_train_test_from_db.py` This makes the files for training on the two part data required for the seq2seq problems. It takes as input a sqlite3 database created with one of the scripts in the root directory of the project. This data is typically extracted from reddit json dumps.
* `model/do_make_vocab.py` This script is executed after the 'do_make_train_test' scripts. It sets up a vocabulary file for the NN model.