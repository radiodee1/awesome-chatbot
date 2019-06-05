# Dataset Preperation and GPT2 Training

* move to the root of the `awesome-chatbot` project.

* Download the movie dialog corpus with the script `./do_make_db_tab_from_cornell_movie.sh` . This will be the basis for your training corpus. 
Alternately you can use reddit dumps using `./do_make_reddit_download.sh` .

* Use `./seq_2_seq/do_split.py` for generating training data for training gpt2.
For convenience the script `./model/do_split_run.sh` appears in this folder. You may have
to move the ouput of the `do_split.py` file into the `./data` folder.

```
$ ./do_split_run.sh --filename ../../rc-movie.txt --to-gpt2 --length=500 --mode=train 
$ cd ../..
$ mv train.* awesome-chatbot/data/.
$ cd awesome-chatbot/model
```


* Make sure you have run the `./do_make_submodule_init.sh` script in the root directory. Make sure you
have looked at the `requirements.txt` folder and you have installed all the `pip3` packages you might need.

* Move to the `model` subfolder.

* See that a checkpoint exists at `../data/tf_gpt2_data/` . You need this tensorflow checkpoint if you are going to do fine-tuning. You can 
get this checkpoint also by running `./model/tf_gpt2_download_model.py` .

* At this point you should be able to run the `./model/torch_pgt2_run_memory.py` file
and test basic chatbot functionality.

* Train the checkpoints further with `./model/tf_gpt2_train.py ` . This
will put a new set of trained checkpoints at `../saved/tf_gpt2_saved/` . 

```
$ ./tf_gpt2_train.py --dataset ../data/train.from --learning_rate=0.00001 --sample_length=25 --run_name=run2 --stop_after=500
```


* Convert the newly trained checkpoints to a pytorch file using `./tf_gpt2_torch_convert.py`. 

* Run the `./torch_gpt2_run_memory.py` script and point the script at the new
converted pytorch file. Use the `--source_file` option to do this. You should see
a change in behaviour that reflects the training that you did.

