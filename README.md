# `awesome-chatbot`
Keras implementation of a chatbot. The basic idea is to start by setting up your training environment as described below and then training with or without autoencoding. The inspiration for this project is the tensorflow NMT project found at the following link: https://github.com/tensorflow/nmt Also, this was inspiring: https://pythonprogramming.net/chatbot-deep-learning-python-tensorflow/ Finally there was a great deep learning youtube series from Siraj Raval. A link for that is [here](https://www.youtube.com/watch?v=t5qgjJIBy9g&index=17&list=PL-pLHOzIUduUSTkdsLkToPdegSbpFJXcX) 
# Organization
The folders and files in the project are organized in the following manor. The root directory of the project is called `awesome-chatbot`. In that folder are sub folders named `data`,  `model`, `raw` and `saved`. There are several script files in the main folder along side the folders mentioned above. These scripts all have names that start with the word `do_` . This is so that when the files are listed by the computer the scripts will all appear together. Below is a folder by folder breakdown of the project.
* `data` This folder holds the training data that the model uses during the `fit` and `predict` operations. The contents of this folder are generally processed to some degree by the project scripts. This pre-processing is described below. This folder also holds the `vocab` files that the program uses for training and inference. The modified word embeddings are also located here.
* `model` This folder holds the python code for the project. Though some of the setup scripts are also written in python, this folder holds the special python code that maintains the keras model. This model is the framework for the neural network that is at the center of this project. There are also two setup scripts in this folder.
* `raw` This folder holds the raw downloads that are manipulated by the setup scripts. These include the GloVe vectors and the Reddit Comments download.
* `saved` This folder holds the saved values from the training process.

Description of the individual setup scripts is included below.
# Suggested Reading - Acknowledgements
* Some basic material on sequence to sequence NMT models came from these sources. The first link is to Jason Brownlee's masterful blog series. The second is to Francois Chollet's Keras blog.
  * https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
  * https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
* Specifically regarding attention decoders and a special hand written Keras layer designed just for that purpose. The author of the layer is Zafarali Ahmed. The code was designed for an earlier version of Keras and Tensorflow. Zafarali's software is provided with the ['GNU Affero General Public License v3.0'](https://github.com/datalogue/keras-attention/blob/master/LICENSE) 
  * https://medium.com/datalogue/attention-in-keras-1892773a4f22
  * https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/
* Pytorch code was originally written by Sean Robertson for the Pytorch demo and example site.
  * http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-intermediate-seq2seq-translation-tutorial-py
# GloVe and W2V Word Embeddings Download
* This link brings you to a page where you can download W2V embeddings that google makes available. At the time of this writing this project does not use w2v embeddings, but uses GloVe instead.
  * https://code.google.com/archive/p/word2vec/
* This link starts a download of the GloVe vectors in the `glove.6B` collection. The download takes a while and uses 823M.
  * http://nlp.stanford.edu/data/glove.6B.zip
# REDDIT Download
* This link starts a download that takes several hours for the Reddit Comments file from November of 2017. The file is several gigabytes.
  * http://files.pushshift.io/reddit/comments/RC_2017-11.bz2
# Scripts For Setup
Here is a list of scripts and their description and possibly their location. You must execute them in order. It is recommended that you install all the packages in the `requirements.txt` file. You can do this with the command `pip3 install -r requirements.txt`
1. `do_make_glove_download.sh` This script is located in the root folder of the repository. It takes no arguments. Execute this command and the GloVe word embeddings will be downloaded on your computer. This download could take several minutes. The file is found in the `raw` folder. In order to continue to later steps you must unpack the file. In the `raw` directory, execute the command `unzip glove.6B.zip`. 
2. `do_make_reddit_download.sh` This script is located in the root folder of the repository. It takes no arguments. Execute this command and the Reddit Comments JSON file will be downloaded on your computer. This download could take several hours and requires several gigabytes of space. The file is found in the `raw` folder. In order to continue to later steps you must unpack the file. In the `raw` directory execute the command `bunzip2 RC_2017-11.bz2`. Unzipping this file takes hours and consumes 30 to 50 gigabytes of space on your hard drive.
3. `do_make_db_from_reddit.py` This script is located in the root folder of the repository. It takes one argument, a specification of the location of the uunpacked Reddit Comments JSON file. Typically you would execute the command as `./do_make_db_from_reddit.py raw/RC_2017-11`. Executing this file takes several hours and outputs a sqlite data base called `input.db` in the root directory or your repository. There should be 5.9 Million paired rows of comments in the final db file. You can move the file or rename it for convenience. I typically put it in the `raw` folder. This python script uses `sqlite3`.
4. `do_make_train_test_from_db.py` This file is not located in the root folder of the repository. It is in the subfolder that the `model.py` file is found in. Execute this file with one argument, the location of the `input.db` file. The script takes several hours and creates many files in the `data` folder that the `model.py` file will later use for training. These data files are also used to create the vocabulary files that are essential for the model.
5. `do_make_vocab.py` This file is located in the directory  that the `do_make_train_test_from_db.py` is found in. It takes no arguments. It proceeds to find the most popular words in the training files and makes them into a list of vocabulary words of the size specified by the `settings.py` file. It also adds a token for unknown words and for the start and end of each sentence. It could take hours to run. It puts a vocabulary list in the `data` folder, along with a modified GloVe word embeddings file.
6. `do_make_rename_train.sh` This file should be called once after the data folder is set up to create some important symbolic links that will allow the `model.py` file to find the training data. If your computer has limited resources this method can be called with a single integer, `n`, as the first argument. This sets up the symbolic links to piont the `model.py` file at the `n`th training file. It should be noted that there are about 80 training files in the `RC_2017-11` download, but these training files are simply copies of the larger training file, called `train.big.from` and `train.big.to`, split up into smaller pieces. When strung together they are identical to the bigger file. If your computer can use the bigger file it is recommended that you do so. If you are going to use the larger file, call the script withhout any arguments. If you are going to use the smaller files, call the script with the number associated with the file you are interested in. This call woudl look like this: `./do_make_rename_train.sh 1`
# Scripts For Train - `do_launch_model.sh`
This is a script for running the `model.py` python file located in the `model` folder. There are several commandline options available for the script. Type `./do_launch_model.sh --help` to see them all. Some options are listed below.
* `--help` This prints the help text for the program.
* `--mode=MODENAME` This sets the mode for the program. It can be one of the following:
  * `train` This is for training the model for one pass of the selected training file.
  * `long` This is for training the model for several epochs on the selected training files. It is the preferred method for doing extended training.
  * `infer` This just runs the program's `infer` method once so that the state of the model's training might be determined from observation.
  * `review` This loads all the saved model files and performs a `infer` on each of them in order. This way if you have several training files you can choose the best.
  * `interactive` This allows for interactive input with the `predict` part of the program.
* `--printable=STRING` This parameter allows you to set a string that is printed on the screen with every call of the `fit` function. It allows the `do_launch_series_model.py` script to inform the user what stage training is at, if for example the user looks at the screen between the switching of input files. (see description of `do_launch_series_model.py` below.)
* `--baename=NAME` This allows you to specify what filename to use when the program loads a saved model file. This is useful if you want to load a filename that is different from the filename specified in the `settings.py` file. This parameter only sets the basename.
* `--autoencode` This option turns on auto encoding during training. It overrides the `model/settings.py` hyper parameter.
* `--train-all` This option overrides the `settings.py` option that dictated when the embeddings layer is modified during training. It can be used on a saved model that was created with embedding training disabled.
# Scripts For Train - `do_launch_series_model.py`
This script is not needed if your computer will run the `--mode=long` parameter mentioned above for the `do_launch_model.sh` script. If your computer has limited memory or you need to train the models in smaller batches you can use this script. It takes no arguments initially. It goes through the training files in the `data` folder and runs the training program on them one at a time. There are two optional parameters for this script that allow you to specify the number of training files that are saved, and also the number of epochs you want the program to perform.
# Hyper-parameters - `model/settings.py`
This file is for additional parameters that can be set using a text editor before the `do_launch_model.sh` file is run.
* `save_dir` This is the relative path to the directory where model files are saved.
* `data_dir` This is the relative path to the directory where training and testing data ate saved.
* `embed_name` This is the name of the embed file that is found in the `data` folder.
* `vocab_name` This is the name of the primary vocabulary list file. It is found in the `data` folder.
* `test_name` This is the name of the test file. It is not used presently.
* `test_size` This is the size of the test file in lines. It is not used.
* `train_name` This is the name of the train file. It is the 'base' name so it doesn't include the file ending.
* `src_ending` This is the filename ending for the source test and training files.
* `tgt_ending` This is the filename ending for the target test and training files.
* `base_filename` This is the base filename for when the program saves the network weights and biases.
* `base_file_num` This is a number that is part of the final filename for the saved weights from the network.
* `num_vocab_total` This number is the size of the vocabulary. It is also read by the `do_make_vocab.py` file. It can only be chhanged when the vocabulary is being created before training.
* `batch_size` Training batch size. May be replaced by `batch_constant`.
* `steps_to_stats` Number representing how many times the `fit` method is called before the stats are printed to the screen.
* `epochs` Number of training epochs.
* `embed_size` Dimensionality of the basic word vector length. Each word is represented by a vector of numbers and this vector is as long as `embed_size`. This can only take certain values. The GloVe download, mentioned above, has word embedding in only certain sizes. These sizes are: 50, 100, 200, and 300.
* `embed_train` This is a True/False parameter that determines whether the model will allow the loaded word vector values to be modified at the time of training.
* `autoencode` This is a True/False parameter that determines whether the model is set up for regular encoding or autoencoding during the training phase.
* `infer_repeat` This parameter is a number higher than zero that determines how many times the program will run the `infer` method when stats are being printed.
* `embed_mode` This is a string. Accepted values are 'mod' and 'normal'. This allows the development of code that will test out different testing scenarios. 'mod' is not supported at the time of this writing. Use 'normal' at all times.
* `dense_activation` There is a dense layer in the model and this parameter tells that layer how to perform its activations. If the value None or 'none' is passed to the program the dense layer is skipped entirely. The value 'softmax' was used initially but produced poor results. The value 'tanh' produces some reasonable results.
* `sol` This is the symbol used for the 'start of line' token.
* `eol` This is the symbol used for the 'end of line' token.
* `unk` This is the symbol used for the 'unknown word' token.
* `units` This is the initial value for hidden units in the first LSTM cell in the model.
* `learning_rate` This is the learning rate for the 'adam' optimizer.
* `tokens_per_sentence` This is the number of tokens per sentence.
* `batch_constant` This number serves as a batch size parameter.