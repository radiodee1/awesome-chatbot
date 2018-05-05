# `awesome-chatbot`
Keras or pytorch implementation of a chatbot. The basic idea is to start by setting up your training environment as described below and then training with or without autoencoding. The inspiration for this project is the tensorflow NMT project found at the following link: https://github.com/tensorflow/nmt Also, this was inspiring: https://pythonprogramming.net/chatbot-deep-learning-python-tensorflow/ Finally there was a great deep learning youtube series from Siraj Raval. A link for that is [here](https://www.youtube.com/watch?v=t5qgjJIBy9g&index=17&list=PL-pLHOzIUduUSTkdsLkToPdegSbpFJXcX)
# Organization
The folders and files in the project are organized in the following manor. The root directory of the project is called `awesome-chatbot`. In that folder are sub folders named `data`,  `model`, `raw` and `saved`. There are several script files in the main folder along side the folders mentioned above. These scripts all have names that start with the word `do_` . This is so that when the files are listed by the computer the scripts will all appear together. Below is a folder by folder breakdown of the project.
* `data` This folder holds the training data that the model uses during the `fit` and `predict` operations. The contents of this folder are generally processed to some degree by the project scripts. This pre-processing is described below. This folder also holds the `vocab` files that the program uses for training and inference. The modified word embeddings are also located here.
* `model` This folder holds the python code for the project. Though some of the setup scripts are also written in python, this folder holds the special python code that maintains the keras model. This model is the framework for the neural network that is at the center of this project. There are also two setup scripts in this folder.
* `bot` This folder is the home of programs that are meant to help the chatbot run. This includes speech-to-text code and speech-recognition code. Ultimately this directory will be the home of a loop of code that monitors audio input from a microphone and decides what to do with it.
* `raw` This folder holds the raw downloads that are manipulated by the setup scripts. These include the GloVe vectors and the Reddit Comments download.
* `saved` This folder holds the saved values from the training process.
* `stats` This folder holds code for creating tables that show training results for various model configuration.

Description of the individual setup scripts is included below.
# Suggested Reading - Acknowledgements
* Some basic material on sequence to sequence NMT models came from these sources. The first link is to Jason Brownlee's masterful blog series. The second is to Francois Chollet's Keras blog.
  * https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
  * https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
* Specifically regarding attention decoders and a special hand written Keras layer designed just for that purpose. The author of the layer is Zafarali Ahmed. The code was designed for an earlier version of Keras and Tensorflow. Zafarali's software is provided with the ['GNU Affero General Public License v3.0'](https://github.com/datalogue/keras-attention/blob/master/LICENSE) 
  * https://medium.com/datalogue/attention-in-keras-1892773a4f22
  * https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/
* Pytorch code was originally written by Sean Robertson for the Pytorch demo and example site. He uses the [MIT license.](https://github.com/spro/practical-pytorch/blob/master/LICENSE)
  * http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#sphx-glr-intermediate-seq2seq-translation-tutorial-py
* Additional Pytorch code was written by Austin Jacobson. A link to his NMT project is included here. He uses the [MIT license.](https://github.com/A-Jacobson/minimal-nmt/blob/master/LICENSE.md)
  * https://github.com/A-Jacobson/minimal-nmt
* Some code was originally written by Yerevann Research Lab. This theano code implements the DMN Network Model. They use the [MIT License.](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/blob/master/LICENSE)
  * https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano
* The original paper on Dynamic Memory Networks, by Kumar et al., can be found here:
  * http://arxiv.org/abs/1506.07285
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
5. `do_make_vocab.py` This file is located in the directory  that the `do_make_train_test_from_db.py` is found in. It takes no arguments. It proceeds to find the most popular words in the training files and makes them into a list of vocabulary words of the size specified by the `settings.py` file. It also adds a token for unknown words and for the start and end of each sentence. If word embeddings are enabled, it will prepare the word embeddings from the GloVe download. The GloVe download does not include contractions, so if it is used no contractions will appear in the `vocab.big.txt` file. The embeddings can be disabled by specifying 'None' for `embed_size` in the `model/settings.py` file. Embeddings can be enabled with some versions of the keras model. The pytorch model is to be used without pre-set embeddings. This script could take hours to run. It puts its vocabulary list in the `data` folder, along with a modified GloVe word embeddings file.
6. `do_make_rename_train.sh` This file should be called once after the data folder is set up to create some important symbolic links that will allow the `model.py` file to find the training data. If your computer has limited resources this method can be called with a single integer, `n`, as the first argument. This sets up the symbolic links to piont the `model.py` file at the `n`th training file. It should be noted that there are about 80 training files in the `RC_2017-11` download, but these training files are simply copies of the larger training file, called `train.big.from` and `train.big.to`, split up into smaller pieces. When strung together they are identical to the bigger file. If your computer can use the bigger file it is recommended that you do so. If you are going to use the larger file, call the script withhout any arguments. If you are going to use the smaller files, call the script with the number associated with the file you are interested in. This call woudl look like this: `./do_make_rename_train.sh 1`
# Scripts For Train - `do_launch_model.sh`
This is a script for running the `model.py` python file located in the `model` folder. There are several commandline options available for the script. Type `./do_launch_model.sh --help` to see them all. Some options are listed below. There is also a `do_launch_babi.sh` file. It works with similar commandline options.
* `--help` This prints the help text for the program.
* `--mode=MODENAME` This sets the mode for the program. It can be one of the following:
  * `train` This is for training the model for one pass of the selected training file.
  * `long` This is for training the model for several epochs on the selected training files. It is the preferred method for doing extended training.
  * `infer` This just runs the program's `infer` method once so that the state of the model's training might be determined from observation.
  * `review` This loads all the saved model files and performs a `infer` on each of them in order. This way if you have several training files you can choose the best.
  * `interactive` This allows for interactive input with the `predict` part of the program.
  * `plot` This runs the review code but also plots a rudimentary graph at the end. This option is only found in the pytorch code.
* `--printable=STRING` This parameter allows you to set a string that is printed on the screen with every call of the `fit` function. It allows the `do_launch_series_model.py` script to inform the user what stage training is at, if for example the user looks at the screen between the switching of input files. (see description of `do_launch_series_model.py` below.)
* `--baename=NAME` This allows you to specify what filename to use when the program loads a saved model file. This is useful if you want to load a filename that is different from the filename specified in the `settings.py` file. This parameter only sets the basename.
* `--autoencode=FLOAT` This option turns on auto encoding during training. It overrides the `model/settings.py` hyper parameter. 0.0 is no autoencoding and 1.0 is total autoencoding.
* `--train-all` This option overrides the `settings.py` option that dictated when the embeddings layer is modified during training. It can be used on a saved model that was created with embedding training disabled.

If you are running the pytorch model, the model will save your last position in the training corpus file whenever it saves the weights. If you want to erase this position and start over in the training file you can erase the 'saved/basename.best.pth.tar' file. (You can also set the 'zero_start' option to True.) This removes the old weights also. To get them back rename the highest saved weights file (or any one of your choosing) to 'basename.best.pth.tar'.
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
* `embed_size` Dimensionality of the basic word vector length. Each word is represented by a vector of numbers and this vector is as long as `embed_size`. This can only take certain values. The GloVe download, mentioned above, has word embedding in only certain sizes. These sizes are: None, 50, 100, 200, and 300. If 'None' is specified then the GloVe vectors are not used. Note: GloVe vectors do not contain contractions, so contractions do not appear in the generated vocabulary files if `embed_size` is not None.
* `embed_train` This is a True/False parameter that determines whether the model will allow the loaded word vector values to be modified at the time of training.
* `autoencode` This is a True/False parameter that determines whether the model is set up for regular encoding or autoencoding during the training phase.
* `infer_repeat` This parameter is a number higher than zero that determines how many times the program will run the `infer` method when stats are being printed.
* `embed_mode` This is a string. Accepted values are 'mod' and 'normal' and only the keras model is effected. This originally allowed the development of code that used different testing scenarios. 'mod' is not supported at the time of this writing. Use 'normal' at all times.
* `dense_activation` There is a dense layer in the model and this parameter tells that layer how to perform its activations. If the value None or 'none' is passed to the program the dense layer is skipped entirely. The value 'softmax' was used initially but produced poor results. The value 'tanh' produces some reasonable results.
* `sol` This is the symbol used for the 'start of line' token.
* `eol` This is the symbol used for the 'end of line' token.
* `unk` This is the symbol used for the 'unknown word' token.
* `units` This is the initial value for hidden units in the first LSTM cell in the keras model. In the pytorch model this is the hidden units value used by both the encoder and the decoder. For the pytorch model GRU cells are used.
* `layers` This is the number of layers for both the encoder and decoder in the pytorch model.
* `learning_rate` This is the learning rate for the 'adam' optimizer. In the pytorch model SGD is used.
* `tokens_per_sentence` This is the number of tokens per sentence.
* `batch_constant` This number serves as a batch size parameter.
* `teacher_forcing_ratio` This number tells the pytorch version of the model exactly how often to use teacher forcing during training.
* `dropout` This number tells the pytorch version of the model how much dropout to use.
* `pytorch_embed_size` This number tells the pytorch model how big to make the embedding vector.
* `zero_start` True/False variable that tells the pytorch model to start at the beginning of the training corpus files every time the program is restarted. Overrides the saved line number that allows the pytorch model to start training where it left off after each restart.

# Raspberry Pi and Speech Recognition
The goal of this part of the project is to provide for comprehensive speech-to-text and text-to-speech for the use of the chatbot when it is installed on a Raspberry Pi. For this purpose we use the excellent google api. The google api 'Cloud Speech API' costs money to operate. If you want to use it you must sign up for Google Cloud services and enable the Speech API for the project. This document will attempt to direct a developer how to setup the account, but may not go into intimate detail. Use this document as a guide, but not necessarily the last word. After everything is set up the project will require internet access to perform speech recognition.

As of this writing the Keras model does not work on the Raspberry Pi because Tensorflow is so difficult to compile for Pi. Tensorflow is the Keras backend that we use in this project.

### PyTorch
An important part of the process of porting this project to the Raspberry Pi is compiling Pytorch for the Pi. At the time of this writing the compiling of Pytorch is possible following the urls below. You do not need to compile Pytorch before you test the speech recognition, but it is required for later steps.
* http://book.duckietown.org/master/duckiebook/pytorch_install.html
* https://gist.github.com/fgolemo/b973a3fa1aaa67ac61c480ae8440e754

### Speech Recognition -- Google
The Google Cloud api is complicated and not all of the things you need to do are covered in this document. I will be as detailed as possible if I can. The basic idea is to install the software on a regular computer to establish your account and permissions. You will need to create a special json authentication file and tell google where to find it on your computer. Then install as much software as possible on the Raspberry Pi along with another special authentication json file. This second file will refer to the same account and will allow google to charge you normally as it would for a regular x86 or x86_64 computer. The speech recognition code in this project should run on the regular computer before you proceed to testing it on the Raspberry Pi.

Install all the recommended python packages on both computers and make sure they install without error. This includes `gtts`, `google-api-python-client`, and `google-cloud-speech`. Install the Google Cloud SDK on the regular computer. The following link shows where to download the SDK. 
* https://cloud.google.com/sdk/docs/

### Resources
You may need to set up a billing account with Google for yourself. Here are some resources for using the Google Cloud Platform.
* https://cloud.google.com/sdk/docs/quickstart-linux See this url for details.
* https://cloud.google.com/speech/docs/quickstart See this location for more google setup info.
* https://console.cloud.google.com/apis/ Try this url and see if it works for you. If you see a dashboard where you can manipulate your google cloud account you are ready to proceed. You want to enable 'Cloud Speech API'.

### Steps for the cloud
* Use Google Cloud Platform Console to create a project and download a project json file. 
  1. Setup a google cloud platform account and project. For a project name I used `awesome-sr`. 
  2. Before downloading the json file, make sure the 'Cloud Speech API' is enabled.
* Download and install the Google-Cloud-Sdk. This package has the `gcloud` command. 
* This download includes the `google-cloud-sdk` folder. Unpack it, and execute the command `./google-cloud-sdk/install.sh`
* You must also restart your terminal.
* I put my project json file in a directory called `/home/<myname>/bin` .
* Use the `gcloud` command to set up your authentication. I used the following: `gcloud auth activate-service-account --key-file=bin/awesome-sr-*.json`
* Use the Google Cloud Platform Console to create a second project json file for the Raspberry Pi. Go to the Downloads folder and identify the Raspberry Pi json file. Transfer the file to the Pi with a command like `scp`.
* Finally you must set up a bash shell variable for both json files so that google can find the json files when you want to do speech recognition. The process for setting up this shell variable is outlined below.

Test google speech recognition with the `bot/game_sr.py` script. The script may be helpful at different times to tell if your setup attempt is working. To execute the script, switch to the `bot/` folder and execute the command `python3 game_sr.py`. 

### Setup Bash Variable
* This guide assumes you are using a linux computer. It also assumes that if you downloaded the json file from the internet and it was stored in your `Downloads` folder, that you have moved it to the root of your home directory. 
* For convenience I made a folder in my home directory called `bin`. This will be the folder for the json file on my  regular computer.
* On the Raspberry Pi I navigated to the `/opt` directory and made a folder called `bot`. I placed the json file at `/opt/bot/`.
* For simplicity I will refer to the json file on my regular computer as `awesome-sr-XXXXXX.json`. In this scheme `awesome-sr` is the name of my project and `XXXXXX` is the hexadecimal number that google appends to the json file name. Because this name is long and the hex digits are hard to type I will copy and paste them when possible as I set up the Bash shell variable.
* Edit the `.bashrc` file with your favorite editor.
* Add the following to the  last line of the `.bashrc` file: `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/json/awesome-sr-XXXXXX.json` A link follows that might be helpful: https://cloud.google.com/docs/authentication/getting-started#setting_the_environment_variable
* Save the changes.
* You must exit and re-enter the bash shell in a new terminal for the changes to take effect. After that you should be able to run the `game_sr.py` file. You will be charged for the service.
* On the Raspberry Pi use the same general technique as above. Edit the `.basshrc` file to contain the line `export GOOGLE_APPLICATION_CREDENTIALS=/opt/bot/awesome-sr-XXXXXX.json` where `XXXXXX` is the hexadecimal label on the json file on the Rapberry Pi. This number will be different from the one on your regular computer.