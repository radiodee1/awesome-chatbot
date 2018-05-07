# Statistics and File Naming:

This page is for results of the various tests in the BABI set and for explaining the naming conventions that we're trying to stick to. The  table may be incomplete.

#### Babi:
There are 20 tests. One at a time we will try to fill in the results that we achieve in the table below. Also below is a list of filenames. Though one model is meant to complete all the tests with the same saved weights, we willl assume that each test is done seperately. At the end we may include a result for testing on all the categories at once.

* `baseline_dmn` This would be a figure for comparrison that was taken from literature on the subject. It does not reflect work done with this project. It is generally used with heavy supervision. [See here.](https://arxiv.org/pdf/1506.07285.pdf)
* `small_embedding` This would be a figure for comparrison. Here the embedding weights only contain words that are in the babi corpus. The embeddings will not be trainable.
* `ii_a` Since the 'babi_ii.py' file is the most complete model it will be the starting base filename for out tests. This column reflects testing done with 100 hidden units and a large set of trainable embeddings. 
* `ii_b` This would reflect 100 hidden units but frozen non-trainable embeddings. Note that this does not mean pre-trained embeddings but rather frozen embeddings with entirely random contents.
* `ii_c` This column would have pre-trained embeddings and 100 units in the hidden layer. Other words would be included in the vocabulary not specific to the question answering task.

The table is included here:

 |   | baseline_dmn | small_embedding | ii_a | ii_b | ii_c | 
|-|-|-|-|-|-|
 | QA1: Single Supporting Fact | 1 | 2 | 3 | 0 | 0 |
 | QA2: Two Supporting Facts | 1 | 2 | 3 | 0 | 0 |
 | QA3: Three Supporting Facts | 0 | 0 | 0 | 0 | 0 |
 | QA4: Two Argument Relations | 0 | 0 | 0 | 0 | 0 |
 | QA5: Three Argument Relations | 0 | 0 | 0 | 0 | 0 |
 | QA6: Yes/No Questions | 0 | 0 | 0 | 0 | 0 |
 | QA7: Counting | 0 | 0 | 0 | 0 | 0 |
 | QA8: Lists/Sets | 0 | 0 | 0 | 0 | 0 |
 | QA9: Simple Negation | 0 | 0 | 0 | 0 | 0 |
 | QA10: Indefinite Knowledge | 0 | 0 | 0 | 0 | 0 |
 | QA11: Basic Coreference | 0 | 0 | 0 | 0 | 0 |
 | QA12: Conjunction | 0 | 0 | 0 | 0 | 0 |
 | QA13: Compound Coreference | 0 | 0 | 0 | 0 | 0 |
 | QA14: Time Reasoning | 0 | 0 | 0 | 0 | 0 |
 | QA15: Basic Deduction | 0 | 0 | 0 | 0 | 0 |
 | QA16: Basic Induction | 0 | 0 | 0 | 0 | 0 |
 | QA17: Positional Reasoning | 0 | 0 | 0 | 0 | 0 |
 | QA18: Size Reasoning | 0 | 0 | 0 | 0 | 0 |
 | QA19: Path Finding | 0 | 0 | 0 | 0 | 0 |
 | QA20: Agent's Motivation | 0 | 0 | 0 | 0 | 0 |

#### Training and Testing:
All results at this time benefit from weak or no supervision during training.
Training benefits from stopping the training process and restarting it with a much lower 'learning_rate' at the end.

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