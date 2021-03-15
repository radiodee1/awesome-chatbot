# `awesome-chatbot`
The goal of this project is to make a Keras, Tensorflow, or Pytorch implementation of a chatbot.
The basic idea is to start by setting up your training environment as described below and then training on various data sets. 
Later we want to use our code to implement a chatbot. This requires finding a suitable data set. 
The inspiration for this project is the tensorflow NMT project found at the following link: [here](https://github.com/tensorflow/nmt) 
Also, this was inspiring: [here](https://pythonprogramming.net/chatbot-deep-learning-python-tensorflow/) 
Finally there was a great deep learning youtube series from Siraj Raval. 
A link for that is [here](https://www.youtube.com/watch?v=t5qgjJIBy9g&index=17&list=PL-pLHOzIUduUSTkdsLkToPdegSbpFJXcX)

# Progress
6/23/18 - The project is in its early stages. No model in this project implements an AI chatbot. That is the goal, but it has not been reached as of yet. Presently the focus is on the babi data set. This implies that we are using pytorch, not keras at this time.

7/10/18 - Fixed a problem regarding the encoder embeddings in the babi_iv.py file. New values were used with every restart. Must return to fix situation where embeddings are not frozen.

7/19/18 - I found this paper. It refers to DMN+, a more advanced Dynamic Memory Network, which can work with the babi data set. This is the link: https://arxiv.org/abs/1603.01417 . They say in the paper that the input module on the basic DMN sometimes experiences overfitting. I followed the example as best as I could and my validation is much better on task 1. At this time I started using the 'en-10k' babi data set. The training portion of this data set is 10,000 questions long.

8/6/18 - I am trying to reproduce the 20 tests from the babi test set. At the same time I'm adding a recurrent GRU decoder to the output of the whole model to see if I can get multi-word output to work. This would help with the resolution of babi task number 19, which employs two word output. The project is still a work in progress.

8/15/18 - I have not been able to finish babi test 19. I have tried adding an output gru to the answer module of the program. My results with that gru are poor.
The program recognizes that the answer must be a pair of cardinal directions, like north, south, east, and west. The program does not, however, identify the correct pair.

9/24/18 - I am still working on multiple word output. WHAT I TRIED: I have placed a dropout function in the code for the recurrent output section of the `babi_recurrent.py` program.
The output of my first test run is promising. I think to start with I will run it with task #1 and then I will try it out on task #19. In other people's code I have noticed that they typically do not
use 50% dropout, but something closer to 30%. I can run task #1 and #19 with lower dropout if I want to later. WHAT WORKED:
The task #1 experiment worked with the recurrent output code. The #19 tests still don't work!

10/6/18 - I've completed some code that runs on hadoop and helps clean input from the reddit download. I have been
keeping it in a separate repository. I will include a link to the github repository here: https://github.com/radiodee1/awesome-hadoop-reddit .

2/19/19 - I stopped working on the babi code and focused on the sequence to sequence code. 
The code originally ran with batches of 50 sentence pairs at a time. 
This code ran but I could not tell if it was training properly. 
I changed the decoder so that while the encoder ran with batches the decoder ran one sentence pair at a time. 
With this code training is very slow and at first I could not tell if it was learning at all. 
I found out though, that this model does learn the rudimentary chatbot task if it is trained long enough. 
I am currently training for several days at a time. 
It turned out that teacher-forcing was very important for this training process. 
I will keep working on this sequence to sequence model and if I have time I will return to the babi code and try to finish task # 19.

3/8/19 - The raspberry pi uses a debian based operating system, so to ensure that the chatbot software is run at boot-up, the `/etc/rc.local` file needs to be edited.
Since the file uses bash styled commands, the change to the file can be made with a single line. `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json && cd /path/to/project/ && ./do_launch_game.sh`
That is the line that must be added to `rc.local`. Also at this time I am experimenting with an open ended vocabulary for `seq_2_seq.py`.

3/17/19 - Code was added to the `./do_launch_game.sh` script so that it will launch the `seq_2_seq.py` model if there exists in the root project folder a file named 'launch'. 
This file does not exist in the github repository, so if you use the repository you must add it yourself.
This is so that the `/etc/rc.local` file can be set up once and the automatic startup of the seq_2_seq model can be disabled on the raspberry pi at the project directory. 

4/9/2019 - The `model` directory has turned into a sandbox of sorts for exploring different neural network implementations. 
I am currently most interested in bert. 
When I am done with this exploration 
I'll spend some time cleaning up the `model` directory.

4/25/2019 - I found some pytorch code on a github site for 'gpt2' that seems to work for me. It doesn't do everything I want but it's a good start.
This is the site: https://github.com/huggingface/pytorch-pretrained-BERT . The version of gpt2 that I'm using is the smaller released version.
I believe the larger version is not available. In any case the model does respond with sentence type output when it is given the same.
Previously I was trying to get a downloaded version of BERT to respond in sentences. That does not seem to work. 
I guess next I'm hoping to fine tune the gpt2 code so that it works better if I can.

5/10/2019 - The code I was using on 4/25 did not work, but I found another repository at https://github.com/graykode/gpt-2-Pytorch that works. 
In fact it works well enough that I've decided to use it in a full blown version of my chatbot. The code, is in Pytorch and also uses the 'huggingface' code from 4/25. 
I've made his github repository a submodule of this project because I use the code like a library. 
This is still all without fine tuning anything. The people at OpenAI who created gpt2 call this sort of implementation 'zero-shot'. 
This means that the model is used right off of the shelf with no fine tuning. This is very interesting. 
I use this 'zero-shot' approach and it works well. At the same time I am finalizing the sequence-to-sequence model, so that I will have two examples for show. 
One will be the best sequence-to-sequence model I have and the other will be the zero-shot gpt2 model. In order to try this code,
after cloning this github repository, run the script named `do_make_submodule_init.sh` .
This will pull the submodule and put the gpt2 data in the right directory.

5/29/2019 - I have tried to re-organize the project folders somewhat.

7/5/2019 - I have added a 'transformer' model which I train from the 'persona' sentence
corpus. It is located in the 'transformer' folder. The model is trained from scratch so
there is no *transfer-learning* going on. The model works on a laptop. Now I should try
to see if I can port the code over to the Raspberry Pi. I do believe, at this writing, that the memory
footprint of the model is small enough for the Raspberry Pi but there are some libraries
that the model requires that need to be ported to the Pi and at this writing I don't know
if that is possible.

8/16/2019 - For the transformer model, I wrote a script that will run the model on a Raspberry Pi. 
It uses a version of the package 'tensorflow_model_server'. 
The version I use for the 'armhf' platform is from the following web site: https://github.com/emacski/tensorflow-serving-arm . 
The transformer model is set up to launch automatically when the pi is plugged in. 
It takes about two minutes to boot up. 
After that it answers some basic questions that a chatbot might be expected to answer. 
It is not as versatile as the GPT2 based model. 
The GPT2 model will not fit on a Raspberry Pi, but the tensorflow transformer model will.

9/9/2019 - I have this goal of running the gpt2 chatbot model on some kind of small board computer. 
I had been considering buying an O-droid. 
Conveniently the raspberry pi 4 was released with a 4 GB memory option. 
I ordered one and installed the pytorch part of the model on it. 
To do this I had to compile pytorch for armv7 specifically for python 3.7. I did this. 
The only thing not tested at this time is speech-to-text and text-to-speech. 
Interestingly the speech libraries work already on the raspberry pi 3B. 
Compiling pytorch was not trivial and took several tries. 
Also interestingly each reply from the chatbot takes ten to fifteen seconds. 
I have yet to determine if this makes the deployed model unusable.

1/5/2020 - I have a version of the gpt-2 model that refers to an aiml file before giving an answer.
This is an experimental python script that is not currently used in any of my Raspberry Pi setups.
The idea is that using a single aiml file you could give the gpt-2 chatbot more specific instructions about how to answer particular questions.
The model answers random questions with the sort of answer you would expect from a model trained on Reddit data,
but answers the questions found in the aiml file with the specific answers in the aiml.

1/6/2020 - I have disgarded my hand-coded version of the sequence to sequence chatbot as it did not produce the desired output.
 I have found and employed a chatbot tutorial that could be found for a while on the Pytorch tutorial site.
 The code was authored by Matthew Inkawhich.
 The code works.
 I contacted Mr. Inkawhich and asked him for instructions on how to cite his work in a paper using latex.
 He was nice enough to help with that.

2/4/2020 - I am removing all babi-type code from the repository.

2/28/2020 - I just spent some time on the gpt2 model and getting it to
accept commands and even do some question and answer type stuff.
It will search the internet for wiki articles and then answer questions about them.
This was fun but not crucial to the overall goals of the project.
It will even launch system apps for you if you ask it to.
I use some aiml to do this.
The aiml is the week link in all of this.

3/9/2020 - Code was written so that Raspberry Pi installations could use indicator lights to show if output is being accepted or if the bot is processing input.
The code was tested with indicator LEDs on the Raspberry Pi.

6/15/2020 - The old readme file was moved to a safe location and this new readme file was created.

# Organization
The folders and files in the project are organized in the following manor. 
The root directory of the project is called `awesome-chatbot`. 
In that folder are sub folders named `data`,  `model`, `raw`, `seq_2_seq`, `transformer`, and `saved`.
There are several script files in the main folder along side the folders mentioned above. 
These scripts all have names that start with the word `do_` . 
This is so that when the files are listed by the computer the scripts will all appear together. 
Below is a folder by folder breakdown of the project.

* `data` This folder holds the training data that the model uses during the `fit` and `predict` operations. The contents of this folder are generally processed to some degree by the project scripts. This pre-processing is described below. This folder also holds the `vocab` files that the program uses for training and inference. The modified word embeddings are also located here.
* `model` This folder holds the python code for the project. 
Though some of the setup scripts are also written in python, this folder holds the special python code that maintains the chatbot model. There are also some setup scripts in this folder.
* `bot` This folder is the home of programs that are meant to help the chatbot run. This includes speech-to-text code and speech-recognition code. Ultimately this directory will be the home of a loop of code that monitors audio input from a microphone and decides what to do with it.
* `raw` This folder holds the raw downloads that are manipulated by the setup scripts. These include the GloVe vectors and the Reddit Comments download.
* `saved` This folder holds the saved values from the training process.
* `classifier` This folder is for BERT and an attempt to create a model that did the chatbot task that at its core was a classifier. The experiment was not successful.
Further BERT experiments can be found here.
* `seq_2_seq` This is for a rnn based sequence to sequence model.
* `transformer` This is for the 'transformer' based chat bot model.  
* `torch_t2t` This experimental directory is for a transformer model written in pytorch. A working model for the chatbot task has not been developed using this code.
* `docker` This folder has scripts for generating docker containers for the project. This was a programming experiment and code from this directory is not used in any of the final installed models.
* `virtualenv` This folder has scripts for working with virtual environments. Some of the scipts are more like templates than working programs.
* `vis` This folder has some code that helps visualize data from the transrormer model and the gpt2 model.
* `experiments` This folder is for experiments.

Description of some of the individual setup scripts is included below.
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
* This paper discusses using a sequence to sequence model ('seq2seq') to make a chatbot:
  * http://arxiv.org/abs/1506.05869v3
* For GPT2 see the following links:
  * https://github.com/graykode/gpt-2-Pytorch
  * https://github.com/huggingface/pytorch-pretrained-BERT


# Scripts For Setup
Here is a list of scripts and their description and possibly their location. It is recommended that you install all the packages in the `requirements.txt` file. You can do this with the command `pip3 install -r requirements.txt`
* `do_make_movie_download.sh` Download movie subtitles corpus.
* `do_make_persona_download.sh` Download the persona corpus.
* `do_make_db_tab_from_cornell_movie.py` Make db file or tab file from movie subtitles corpus. If you make a tab file, separate parts of the file with `do_split.py`. This script is found in the `seq_2_seq` folder.
* `do_make_tab_file_from_persona.py` Make a tab file from the persona corpus.
* `do_make_train_test_from_db.py` This file is not located in the root folder of the repository. It is in the subfolder called `seq_2_seq`. Execute this file with one argument, the location of the `input.db` file. The script takes several hours and creates many files in the `data` folder that the individual `seq_2_seq` file will later use for training. These data files are also used to create the vocabulary files that are essential for the model. The `transformer` model, based on the tensorflow t2t repository, can also use these files.
* `do_make_vocab.py` This file is located in the directory  that the `do_make_train_test_from_db.py` is found in. It takes no arguments. It proceeds to find the most popular words in the training files and makes them into a list of vocabulary words of the size specified by the `settings.py` file. It also adds a token for unknown words and for the start and end of each sentence. If word embeddings are enabled, it will prepare the word embeddings from the GloVe download. The GloVe download does not include contractions, so if it is used no contractions will appear in the `vocab.big.txt` file. The embeddings can be disabled by specifying 'None' for `embed_size` in the `model/settings.py` file. Embeddings can be enabled with some versions of the keras model. The pytorch model is to be used without pre-set embeddings. This script could take hours to run. It puts its vocabulary list in the `data` folder, along with a modified GloVe word embeddings file.
* `do_make_rename_train_valid_test.sh` This script sets up some links for the corpus files.

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

### One-Liner for Shutdown Button `/etc/rc.local`
This runs the shutdown script on the Pi. You must install the physical button on pins #05 and #20.
* `sudo python /home/pi/workspace/awesome-chatbot/shutdown.py &`

If you intend to use a virtual environment, you must activate it in this line.

* `export VIRTUALENVWRAPPER_PYTHON=$(which python3.7) && source $(which virtualenvwrapper.sh) && workon chatbot37 && sudo python /home/pi/workspace/awesome-chatbot/shutdown.py &`

### One-Liner for Program Launch `/etc/rc.local`
On the raspberry pi there is a file called `/etc/rc.local` that is launched with every reboot. Use this file to launch the chatbot/smart-speaker on startup.

* `su pi  -c 'cd /home/pi/workspace/awesome-chatbot/ && GOOGLE_APPLICATION_CREDENTIALS=/home/pi/bin/awesome-sr-xxxxxx.json ./do_launch_game_s2s.sh'`

With virtualenv:

* `su pi  -c 'export VIRTUALENVWRAPPER_PYTHON=$(which python3.7) && source $(which virtualenvwrapper.sh) && workon chatbot37 && cd /home/pi/workspace/awesome-chatbot/ && GOOGLE_APPLICATION_CREDENTIALS=/home/pi/bin/awesome-sr-xxxxxx.json ./do_launch_game_s2s.sh'`

### One-Liner for `start_test.py` Google Cloud loading in `/etc/rc.local`

* `su pi  -c 'GOOGLE_APPLICATION_CREDENTIALS=/home/pi/bin/awesome-sr-xxxxxx.json /home/pi/workspace/awesome-chatbot/start_test.py'`
* Place this line first before all others.

With virtualenv:

* `su pi  -c 'export VIRTUALENVWRAPPER_PYTHON=$(which python3.7) && source $(which virtualenvwrapper.sh) && workon chatbot37 && GOOGLE_APPLICATION_CREDENTIALS=/home/pi/bin/awesome-sr-xxxxxx.json /home/pi/workspace/awesome-chatbot/start_test.py'`
* Place this line first before all others.

### Docker for ARMv7

Run these commands on the raspberry pi. This is necessary for the Transformer model that uses Tensorflow. The GPT2 model uses Pytorch.

```
$ sudo apt-get update
$ sudo apt-get upgrade
$ curl -fsSL test.docker.com -o get-docker.sh 
$ sh get-docker.sh

$ sudo usermod -aG docker $USER
```

Log out and then in again to use docker.

### WIKI SEARCH

You need two credentials stored in files. The files are kept in the `/home/<username>/bin/` folder.
One is the `api_key.txt` file. One is the `cse_id.txt` file. Wiki search will only
work in the Docker version of the project.
* API Key - https://cloud.google.com/docs/authentication/api-keys - place generated key in `~/bin/api_key.txt` file.
* Custom Search Engine ID - https://cse.google.com/cse/all - place generated ID in `~/bin/cse_id.txt` file.
* Getting Started Doc - https://github.com/googleapis/google-api-python-client/blob/master/docs/start.md

### Docker for AMD64

With an update to a version of Ubuntu, Python 3.7 was replaced with a later version and scripts for this project stopped working.
To use this project with Docker, follow the commands below. This should work on any linux amd64 operating system with Docker.
* Install docker on your machine
* `git pull` this project to the directory of your choice.
* `cd` into the project directory.
* `cd` into the `docker` folder.
* `./do_build_amd64.sh` to build the docker container.
* After the build process the script will automatically start the launch script called `./do_launch_amd64.sh`. You will be presented with a bash prompt. This will be inside the Docker container.
* Train networks or run programs to use pre-trained networks from the `bash` prompt.
* Exit `bash` with the `exit` command.
* You can change or add to the contents of the project directory as you wish before and after launching Docker. The contents of the folder are mounted in the Docker context.
* Run `./do_launch_amd64.sh` any time to start the image after build.

### Jetson Nano Headless setup
Flash the OS to the sd card.
Use a wire USB-A male to micro USB male to connect to the host machine for the first time.
Use the command below to setup the Nano for the first time headless.

`sudo screen /dev/ttyACM0 115200`

The Nano runs python 3.6 in the aarch64 environment. 
Launch these scripts in order to setup an environment for the gpt2 small model.
You will probably need to install sentencepiece from source before installing `transformers`.

```
$ sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
$ git clone https://github.com/google/sentencepiece.git
$ cd sentencepiece
$ mkdir build
$ cd build
$ cmake ..
$ make -j $(nproc)
$ sudo make install
$ sudo ldconfig -v
```

continue
```
## install sentencepiece first from source here! ##
$ ./do_make_tf_apt_aarch64.sh
$ sudo python3 -m pip install -r requirements.aarch64.txt
$ ./do_make_submodule_init.sh
```

Install Pytorch 1.5 for python3.6 from the following forum site.

Site: https://forums.developer.nvidia.com/t/pytorch-for-jetson-nano-version-1-5-0-now-available/72048

```
curl https://nvidia.box.com/shared/static/3ibazbiwtkl181n95n9em3wtrca7tdzp.whl -o  torch-1.5.0-cp36-cp36m-linux_aarch64.whl 
```
Then you can test the model at the command line.
This does not ensure text-to-speech or speech-to-text is working.

Some useful pulseaudio commands:

```
alsamixer ## <-- use F6 to find your USB audio
pacmd list-sources
pacmd list-sinks
pacmd set-default-source 0 ## <-- 0 is the microphone's index number from the list.
pacmd set-default-sink 0 ## <-- this will be some number other than 0
```

## Pulseaudio as service:
```
# save as a file at: /etc/systemd/system/pulseaudio.service
# then execute:
# systemctl enable pulseaudio
# systemctl start pulseaudio

[Unit]
Description=PulseAudio Daemon

[Install]
WantedBy=multi-user.target

[Service]
Type=simple
PrivateTmp=true
ExecStart=/usr/bin/pulseaudio  
```

## OPENAI

Place Authentication code in file called: `~/bin/awesome-chatbot-openai.txt`

This file should contain the code issued by OpenAi for the Beta.