This is a preparation area for raw data. To use this data follow the 
instructions below.

# Using reddit corpus with gpt-2

The gpt2 model will fine-tune with either the movie or the reddit corpus.
It takes single line input from a single input file. Here we will discuss fine-tuning with
the reddit corpus, but using the movie download is similar.

The reddit corpus is large. There is a github project that uses hadoop to process
the reddit corpus for the seq_2_seq model. This project processes the entire
corpus. For gpt2 a much smaller segment of the corpus is used and this makes the hadoop
code unnecessary. The remainder of this explanation assumes you only want 500 or so lines
of training data to fine-tune gpt2.

* Go to the root of the `awesome-chatbot` folder.
* Execute the `do_make_reddit_download.sh` script. This script downloads a month of reddit.
It will take a long time.
* Execute the `do_make_db_from_reddit.py` script. There are two useful parameters. The first is the
name of the unzipped reddit download. That will be found in the `raw` directory. The second is the 
desired length of the output database file. In this case you might use a number like 500.
* Go to the `seq_2_seq` folder.
* Execute the `do_make_train_test_from_db.py`. Make sure to add the `--basename`, `--length`, and `--to-gpt2` flags. 
There will be one output file in the `raw` folder. The file will be named 'chat_reddit_tab.txt'.

# Using movie corpus with seq_2_seq

Experimentation shows the seq_2_seq models work best with the Cornell Movie Database download.

* Go to the root of the `awesome-chatbot` folder.
* Execute the `do_make_movie_download.sh` script.
* Execute the `do_make_db_tab_from_cornell_movie.py` script. This will make a 
sqlite3 database or a tab delimited text file depending on 
which options you use. If you use the database file it will be placed in the `raw` folder with the 
name `input_movie.db` .

* If you use the movie db option, move to the `seq_2_seq` folder and execute the 
`do_make_train_test_from_db.py` script. This will copy 'train', 'test', and 'validate' files to the 
`data` folder.

* If you use the text output from the Cornell Movie download, you can generate 'train', 'test' and 'validate' 
files using the `do_split.py` script. It is located in the `seq_2_seq` folder. 

* Make sure you  copy the groups of files called 'train.big', 'test.big', and 'validate.big' to the `data` folder.

* Go back to the root directory for the `awesome-chatbot` project. Run the `do_make_rename_train_valid_test.sh` script 
with the word 'big' as the only parameter.

* Go to the `seq_2_seq` folder and start your training run.