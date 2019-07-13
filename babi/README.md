# Statistics and File Naming:

This page is for results of the various tests in the BABI set and for explaining the naming conventions that we're trying to stick to. The  table may be incomplete.

#### Babi:
There are 20 tests. One at a time we will try to fill in the results that we achieve in the table below. Also below is a list of filenames. Though one model is meant to complete all the tests with the same saved weights, we will assume that each test is done separately. At the end we may include a result for testing on all the categories at once.

The babi data set is downloaded in the form of 20 categories. There is the basic babi english data set and the 10k data set. We will use the 10k data set. Each category has 10,000 training examples and 1000 testing examples. For our purposes we have divided up each category into 10,000 training examples along with 500 validation set examples and 500 testing set examples. Scores may be filed in pairs. If the score is filed in pairs, the first score in the pair is the validation set score and the second is the test set score.

* `baseline_dmn` This would be a figure for comparison that was taken from literature on the subject. It does not reflect work done with this project. [See here](https://arxiv.org/pdf/1506.07285.pdf) and [here.](https://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/#initial-experiments)
* `dmn_plus` Results from the DMN+ paper. [See here.](https://arxiv.org/abs/1603.01417)
* `small` Here the embedding weights only contain words that are in the babi corpus. The embeddings will not be trainable. The word vectors would come from the glove download. There are 158 words used in the babi corpus, and with three special tokens ('sol', 'eol', and 'unk') there are 161 vocabulary words. This would be the classic configuration for the babi test.
* `small_tr` This would be the same small set of embeddings, but not using embedding vectors from the glove download. Embeddings are not pre-trained and are trained by the pytorch model.
* `GPT2` This column reflects testing done with the GPT2 architecture. These results are collected using the `tf_gpt2_train_babi.py` script in the `experements` folder. 

The table is included here:

 |   | baseline_dmn | dmn_plus | small | small_tr | GPT2 |
|-|:-:|:-:|:-:|:-:|:-:|
 | QA1: Single Supporting Fact | 100 | 100 | 100/100 | 100/100 | 100 |
 | QA2: Two Supporting Facts | 98.2 | 99.7 | 0 | 0 | 96.0 |
 | QA3: Three Supporting Facts | 95.2 | 98.9 | 17.20/18.60 | 0 | 38.18 * |
 | QA4: Two Argument Relations | 100 | 100 | 100/100 | 100/100 | 100 |
 | QA5: Three Argument Relations | 99.3 | 99.5 | 99.40/99.80 | 99.60/99.80 | 97.8 |
 | QA6: Yes/No Questions + | 100 | 100 | 100/100 | 100/100 | 98.4 |
 | QA7: Counting | 96.9 | 97.6 | 97.80/97.40 | 98.80/98.60 | 98.6 |
 | QA8: Lists/Sets | 96.5 | 100 | 99.40/100 | 100/100 | 98.8 |
 | QA9: Simple Negation + | 100 | 100 | 98.20/97.20* | 100/99.40 | 97.0 |
 | QA10: Indefinite Knowledge + | 97.5 | 100 | 99.40/99.00 | 99.60/100 | 96.6 |
 | QA11: Basic Coreference | 99.9 | 100 | 100/100 | 100/100 | 97.6 |
 | QA12: Conjunction | 100 | 100 | 100/100 | 100/100 | 99.4 |
 | QA13: Compound Coreference | 99.8 | 100 | 99.80/100 | 100/100 | 95.8 |
 | QA14: Time Reasoning | 100 | 99.8 | 97.20/94.60 | 99.00/99.20 | 87.0 |
 | QA15: Basic Deduction | 100 | 100 | 100/100 | 100/100 | 0 |
 | QA16: Basic Induction | 99.4 | 54.7 | 48.20/50.60 | 52.60/58.40 | 0 |
 | QA17: Positional Reasoning + | 59.6 | 95.8 | 59.20/57.00 | 58.00/59.40 | 0 |
 | QA18: Size Reasoning + | 95.3 | 97.9 | 91.60/89.40 | 91.60/89.40 | 0 |
 | QA19: Path Finding ** | 34.5 | 100 | xx/xx | xx/xx | 97.3 |
 | QA20: Agent's Motivation | 100 | 100 | 100/100 | 100/100 | 0 |
 | Average | 93.605 | 0 | 0 | 0 | 0 |

_*_ -- these results may need to be revisited.

_+_ -- yes/no or yes/no/maybe answers.

_**_ -- Test 19 uses two-word output. See note below.

#### Training and Testing:
All results at this time benefit from weak or no supervision during training. It has been noted in the DMN+ paper that for most models compared, the results for some tests are always 100%. These would be tests 1, 4, 11, 12, 13, 15, and 19.  [See here.](https://arxiv.org/abs/1603.01417)

#### This Directory:

This is a list of the models from the project.
The original project was aimed at making a neural network chatbot.
It was felt that a seq2seq translation model might function as a chatbot if both languages were english.
Later the DMN architecture became interesting.
* `seq_2_seq/babi.py` This is the current version of the code meant to work with the babi dataset.
* `seq_2_seq/settings.py` This file holds all the hyper parameters for all the models from various versions of the project.
* `seq_2_seq/tokenize_weak.py` This file holds some common code for tokenizing sentences. It is used by all the models.

#### Utility Scripts in `seq_2_seq`:
* `seq_2_seq/do_make_train_test_from_babi.py` This must be run before executing any training on the babi set. This script also allows you to select which babi challenge you want to work on.
* `seq_2_seq/do_make_train_test_from_db.py` This makes the files for training on the two part data required for the seq2seq problems. It takes as input a sqlite3 database created with one of the scripts in the root directory of the project. This data is typically extracted from reddit json dumps.
* `seq_2_seq/do_make_vocab.py` This script is executed after the 'do_make_train_test' scripts. It sets up a vocabulary file for the NN model.

#### Two Word Output: `babi_recurrent.py` -- Test 19
8/10/18

To address the problem of two word output the babi.py program was outfitted with a recurrent output module.
The module consisted of a two layer gru. For the babi test 19 we used the specially outfitted program.
This is one of the reasons we don't average our results. They are not all produced using the same software.

The results for test 19 are terrible. If we come across material on line that would improve our results we will attempt to employ it.
Until such time these are the results we will document for test 19.
