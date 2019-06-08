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
* Process the chat_reddit_tab.txt file with the `do_split.py` script from the `seq_2_seq` folder. You should
use the `--to-gpt2` option. This will put 'Q: ' and 'A: ' character strings before every
question and answer, but it will leave the sentences in the same line separated by a tab.
* You should at this point be able to fine tune the gpt2 model with the reddit corpus. Train the checkpoints further with `./model/tf_gpt2_train.py ` . This
will put a new set of trained checkpoints at `../saved/tf_gpt2_saved/` . 

```
$ ./tf_gpt2_train.py --dataset ../data/train.from --learning_rate=1e-6 --sample_length=25 --run_name=run2_reddit --stop_after=500
```


* Convert the newly trained checkpoints to a pytorch file using `./tf_gpt2_torch_convert.py`. 

```
$ ./tf_gpt2_torch_convert.py ../saved/tf_gpt2_saved/run2_reddit/model-30.data-00000-of-00001 ../saved/
```

* Run the `./torch_gpt2_run_memory.py` script and point the script at the new
converted pytorch file. Use the `--source_file` option to do this. You should see
a change in behaviour that reflects the training that you did.

```
$ ./torch_gpt2_run_memory.py --source_file ../saved/pytorch_model.bin 
```

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

* Make sure you copy the groups of files called 'train.big', 'test.big', and 'validate.big' to the `data` folder.

* Make sure you generate vocabulary files in the `data` folder. To do this you use the `do_make_vocab.py` script.

```
./do_make_vocab.py --basefile ../data/train.* --babi --contractions --both-files --limit=15000
```

* Go back to the root directory for the `awesome-chatbot` project. Run the `do_make_rename_train_valid_test.sh` script 
with the word 'big' as the only parameter.

* Go to the `seq_2_seq` folder and start your training run. The start of a typical run might look like this:

```
./seq_2_seq.py \
--mode=long \
--load-babi \
--load-recurrent \
--units=300 \
--length=15 \
--lr=0.001 \
--dropout=0.3 \
--skip-unk \
--hide-unk \
--basename test_s2s_d300_v15000_length15
```