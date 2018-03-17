# awesome-chatbot
Keras implementation of a chatbot

# Suggested Reading
* https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
* https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
* https://medium.com/datalogue/attention-in-keras-1892773a4f22

# GloVe and W2V Word Embeddings Download
* https://code.google.com/archive/p/word2vec/
* http://nlp.stanford.edu/data/glove.6B.zip

# REDDIT Download
* http://files.pushshift.io/reddit/comments/RC_2017-11.bz2

# Scripts For Setup
Here is a list of scripts and their description and possibly their location. You must execute them in order.
1. `do_make_glove_download.sh` This script is located in the root folder of the repository. It takes no arguments. Execute this command and the GloVe word embeddings will be downloaded on your computer. This download could take several minutes. The file is found in the `raw` folder. In order to continue to later steps you must unpack the file. In the `raw` directory, execute the command `unzip glove.6B.zip`. 
2. `do_make_reddit_download.sh` This script is located in the root folder of the repository. It takes no arguments. Execute this command and the Reddit Comments JSON file will be downloaded on your computer. This download could take several hours and requires several gigabytes of space. The file is found in the `raw` folder. In order to continue to later steps you must unpack the file. In the `raw` directory execute the command `bunzip2 RC_2017-11.bz2`. Unzipping this file takes hours and consumes 30 to 50 gigabytes of space on your hard drive.
3. `do_make_db_from_reddit.py` This script is located in the root folder of the repository. It takes one argument, a specification of the location of the uunpacked Reddit Comments JSON file. Typically you would execute the command as `./do_make_db_from_reddit.py raw/RC_2017-11`. Executing this file takes several hours and outputs a sqlite data base called `input.db` in the root directory or your repository. There should be 5.9 Million paired rows of comments in the final db file. You can move the file or rename it for convenience. I typically put it in the `raw` folder. This python script uses `sqlite3`.
4. `do_make_train_test_from_db.py` This file is not located in the root folder of the repository. It is in the subfolder that the `model.py` file is found in. Execute this file with one argument, the location of the `input.db` file. The script takes several hours and creates many files in the `data` folder that the `model.py` file will later use for training. These data files are also used to create the vocabulary files that are essential for the model.
5. `do_make_vocab.py` This file is located in the directory  that the `do_make_train_test_from_db.py` is found in. It takes no arguments. It proceeds to find the most popular words in the training files and makes them into a list of vocabulary words of the size specified by the `settings.py` file. It also adds a token for unknown words and for the start and end of each sentence. It could take hours to run. It puts a vocabulary list in the `data` folder, along with a modified GloVe word embeddings file.
6. `do_make_rename_train.sh` This file should be called once after the data folder is set up to create some important symbolic links that will allow the `model.py` file to find the training data. If your computer has limited resources this method can be called with a single integer, `n`, as the first argument. This sets up the symbolic links to piont the `model.py` file at the `n`th training file. It should be noted that there are about 80 training files in the `RC_2017-11` download, but these training files are simply copies of the larger training file, called `train.big.from` and `train.big.to`, split up into smaller pieces. When strung together they are identical to the bigger file. If your computer can use the bigger file it is recommended that you do so. If you are going to use the larger file, call the script withhout any arguments. If you are going to use the smaller files, call the script with the number associated with the file you are interested in. This call woudl look like this: `./do_make_rename_train.sh 1`
# Scripts For Train
