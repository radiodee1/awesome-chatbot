## Files

* `do_make_train_test_from_db.py` - runnable file to make train and test files found in `../data/`
* `do_make_vocab.py` - runnable file for creating vocab file found in `../data/`
* `do_split.py` - runnable file for processing train and test files found in `../data/`
* `README.md` - this file.
* `seq_2_seq.py` - non-working hand coded example.
* `seq_2_seq_tutorial.py` - working Inkawhich tutorial file.
* `settings.py` - non-runnable hyper-parameter file.
* `tokenize_weak.py` - non runnable tokenizing function.

## Run `seq_2_seq` Folder Models

* This code runs but produces no usable output.

```
./seq_2_seq.py --mode long --basename somename_01 --hide-unk --skip-unk --lr 0.002 --length 25 --load-recurrent --units 500 
```

* This code runs and uses the Inkawhich tutorial. It does produce usable output.

```
./seq_2_seq_tutorial.py --mode long --build-train-data --train
```

* This line uses the 'tutorial' code in inference mode.
```
./seq_2_seq_tutorial.py --mode interactive --iter 4000
```
---

# `seq_2_seq.py` Work In Progress

> This section of the README doc is for future experimentation. Presently the code does not work at a length of execution of 4,000 iterations. Possibly this will work in the future given further experimentation, but it does not work now.

The object is to try to run the 'seq_2_seq.py' model with 500 hidden units for an exceedingly long time and then see what the model converges on.
Several other programs need to be run to set up a virtual environment for this experiment and this README documents that process.
This will take several days of computation and uses no gpu parallelization. We use the linux computer platform.

1. Move to the directory where you do your work. Here work is done in a folder called 'workspace'. Then we download the git repository with the 'pull' command.
```
cd workspace
git clone https://github.com/radiodee1/awesome-chatbot.git
```

2. Here the instructions deviate for the setup of the proper python3.6 environment. Again we move to the 'workspace' folder. Then we download the source for python 3.6, build the source, and install it using the 'altinstall' flag. After that, python3.6 will be available on your system.

```
sudo apt-get install libssl-dev libbz2-dev libffi-dev libsqlite3-dev sqlite3 # other dev packages may be required
wget --no-check-certificate  https://www.python.org/ftp/python/3.6.15/Python-3.6.15.tgz
tar xvzf Python-3.6.15.tgz 
cd Python-3.6.15
./configure --enable-optimizations --enable-loadable-sqlite-extensions
sudo make 
sudo make altinstall
```

3. Move to the 'awesome-chatbot' folder and then to the 'virtualenv' folder. Then you want to create the virtual environment for the project. This is done by sourcing the shell script in the folder. You know that the command works when you see the little 'chatbot36' in the terminal prompt.
```
cd workspace/awesome-chatbot/virtualenv
. ./do_make_virtualenv_setup36.sh
```

4. Next the python and apt/system software is installed. For the python software you must ensure that you are working in the virtualenvironment for the project. This need not be the 'virtualenv' folder, but it does need to be the same terminal that you used to set up the environment. Again, this would have a 'chatbot36' title in the terminal prompt.

```
# move to chatbot folder

cd workspace/awesome-chatbot/

# execute command for apt install of python and tensorflow packages

./do_make_apt_amd64.sh

# NOTE: the script above installs some tensorflow material that is not needed for the seq_2_seq experiment.

pip3 install -r requirements.amd64.txt

# install needed python requirements. Additional python3.6 requirements may be needed at a later time!!
```
5. Move to the proper directory and download the corpus files.

```
cd workspace/awesome-chatbot
./do_make_movie_download.sh
./do_make_unpack_text.sh
```
6. Make a special file for the corpus text. This will be a first step twords processing the text.
```
cd workspace/awesome-chatbot
do_make_db_tab_from_cornell_movie.py \
	./raw/cornell\ movie-dialogs\ corpus/movie_lines.txt \
	--text-file

# this makes a file called 'movie_lines.txt.tab.txt'
```
7. Now modify the generated text file with the 'do_split.py' file from the 'seq_2_seq' folder.

```
cd seq_2_seq/
./do_split.py \
	--filename \
	../raw/cornell\ movie-dialogs\ corpus/movie_lines.txt.tab.txt \
	--start 0 \
	--length 0 \
	--pairs

# this makes two files called 'train.from' and 'train.to'
```
8. Move the two generated files to the 'data' folder. Create 'saved' folder.
```
cd ../raw/cornell\ movie-dialogs\ corpus/
mv train.from train.to ../../data/.
cd ../..
mkdir saved
```
9. Go back to the 'seq_2_seq' folder and make the vocab files needed by using the following python script.
```
cd ../../seq_2_seq/
./do_make_vocab.py \
	--bsefile ../data/train.from \
	--limit 4500 \
	--both-files \

# you might want to experiment and include the flag '--order' above.
```
10. Then you can start the actual experiment.
```
./seq_2_seq.py \
	--mode long \
	--basename local_test_005 \
	--hide-unk \
	--skip-unk \
	--lr 0.002 \
	--length 10 \
	--load-recurrent \
	--units 500 \
	--dropout 0.2 \
	--teacher-forcing 0.8 \
	--add-eol \
	--stop 4000
```
