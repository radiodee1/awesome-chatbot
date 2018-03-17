# awesome-chatbot
Keras implementation of a chatbot

# Suggested Reading
* https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
* https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

# GloVe and W2V Word Embeddings Download
* https://code.google.com/archive/p/word2vec/
* http://nlp.stanford.edu/data/glove.6B.zip

# REDDIT Download
* http://files.pushshift.io/reddit/comments/RC_2017-11.bz2

# Scripts For Setup
Here is a list of scripts and their description and possibly their location. You must execute them in order.
1. `do_make_glove_download.sh` This script is located in the root folder of the repository. It takes no arguments. Execute this command and the GloVe word embeddings will be downloaded on your computer. This download could take several minutes. The file is found in the `raw` folder. In order to continue to later steps you must unpack the file. In the `raw` directory, execute the command `unzip glove.6B.zip`. 
2. `do_make_reddit_download.sh` This script is located in the root folder of the repository. It takes no argumentss. Execute this command and the Reddit Comments JSON file will be downloaded on your computer. This download could take several hours and requires several gigabytes of space. The file is found in the `raw` folder. In order to continue to later steps you must unpack the file. In the `raw` directory execute the command `bunzip2 RC_2017-11.bz2`. Unzipping this file takes hours and consumes 30 to 50 gigabytes of space on your hard drive.
3. `do_make_db_from_reddit.py` This script is located in the root folder of the repository. It takes one argument, a specification of the location of the uunpacked Reddit Comments JSON file. Typically you would execute the command as `./do_make_db_from_reddit.py raw/RC_2017-11`. Executing this file takes several hours and outputs a sqlite data base called `input.db` in the root directory or your repository. There should be 5.9 Million paired rows of comments in the final db file. You can move the file or rename it for convenience. I typically put it in the `raw` folder. This python script uses `sqlite3`.
4. `do_make_train_test_from_db.py` 
# Scripts For Train
