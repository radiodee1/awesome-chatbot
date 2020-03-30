# Movie Dataset Preperation and GPT2 Training

* move to the root of the `awesome-chatbot` project.

* Download the movie dialog corpus with the script `./do_make_db_tab_from_cornell_movie.sh` . This will be the basis for your training corpus. 
Alternately you can use reddit dumps using `./do_make_reddit_download.sh` .

* Use `./seq_2_seq/do_split.py` for generating training data for training gpt2.
For convenience the script `./model/do_split_run.sh` appears in this folder. You may have
to move the output of the `do_split.py` file into the `./data` folder.

```
$ ./do_split_run.sh --filename ../../rc-movie.txt --to-gpt2 --length=500 --mode=train.big --zip=chat_gpt2_movie
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

```
$ ./torch_gpt2_run_memory.py 
```


* Train the checkpoints further with `./model/tf_gpt2_train.py ` . This
will put a new set of trained checkpoints at `../saved/tf_gpt2_saved/` . 

```
$ ./tf_gpt2_train.py --dataset ../data/train.from --learning_rate=0.00001 --sample_length=25 --run_name=run2 --stop_after=500
```


* Convert the newly trained checkpoints to a pytorch file using `./tf_gpt2_torch_convert.py`. 

```
$ ./tf_gpt2_torch_convert.py ../saved/tf_gpt2_saved/run2/model-30.data-00000-of-00001 ../saved/
```

* Run the `./torch_gpt2_run_memory.py` script and point the script at the new
converted pytorch file. Use the `--source_file` option to do this. You should see
a change in behaviour that reflects the training that you did.

```
$ ./torch_gpt2_run_memory.py --source_file ../saved/pytorch_model.bin 
```
---

## File Description:

* `do_launch_tensorboard.sh` - Script to make running tensorboard easier.
* `do_split_run.sh` - Script to run 'do_split.py' found in 'seq_2_seq' folder.
* `nmt_aiml_commands.py` - Subscript for running aiml in gpt2 scripts.
* `nmt_wiki_commands.py` - Subscript for running wiki searches in gpt2 scripts.
* `settings.py` - file that contains some hyper-parameters.
* `tf_gpt2/` - folder with Tensorflow GPT2 repository.
* `tf_gpt2_download_model.py` - Script for downloading gpt2 files.
* `tf_gpt2_torch_convert.py` - Convert tensorflow gpt2 to pytorch gpt2 format.
* `tf_gpt2_train.py` - Train a tensorflow gpt2 image after download.
* `tokenize_weak.py` - Subscript for tokenizing input strings.
* `torch_gpt2/` - folder with Pytorch GPT2 repository.
* `torch_gpt2_run_memory_common.py` - User runnable GPT2 chatbot script.
* `torch_gpt2_run_memory.py` - User runnable GPT2 chatbot script.
* `torch_gpt2_run_memory_substitute_aiml_lrg.py` - User runnable GPT2 chatbot script. Imports `nmt_wiki_commands.py`. This file searches on the internet.
* `torch_gpt2_run_memory_substitute_aiml_sm.py` - User runnable GPT2 chatbot script. Imports `nmt_wiki_commands.py`. This file searches on the internet.
* `torch_gpt2_run_memory_trained.py` - User runnable GPT2 chatbot script. Specifically for files trained after download.
