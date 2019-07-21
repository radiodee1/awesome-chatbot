## Transformer

The goal here is to create a chatbot or smart speaker with the transformer
architecture. Much of this code relies on the Tensor2Tensor project from 
Google. The training files in this directory do not use the same training
corpus files that are used by the babi code or the seq2seq code. Instead they
use a single file called `../data/raw.txt`. This file is generated from the 
'Persona' data set.

* `tf_t2t_train_chat.py` - This is the script for training the transformer with
the 'Persona' data. There are some important flags. They are `--name`, `--train` and
`--test`. You want to specify the name with every run. The default is 'chat' but can
easily be replaced with something like 'chat_01' or 'chat_02'. When you are 
training you should use the '--train' flag. At regular intervals you want to
open a new terminal and run the code with the '--test' flag. This will show
you what your output looks like subjectively. When you do this you don't want
to stop the original training script. Every time you stop the script tensorflow
starts reading your corpus files from the beginning.  

* `tf_t2t_train_serve.py` - This is the script for converting the trained model
into a servable export. You run this model with the '--name' flag. Use the same
name as you did in your training run. Then you use the '--export' flag for
converting the saved training output to a form that tensorflow can serve. Then
you can test your servable file using a second run with the '--query' flag. Again
here you want to specify the name that you gave the model during training.

* `tf_t2t_train_run.py` - This script can be run from the command line but is
mostly meant for the use of the chatbot scripts that do speech recognition. This
script is imported by the code in the `../bot/game.py` script. Then
after it is imported it acts as a interface between the 'game.py' file and
the tensor2tensor transformer model that you create with the `tf_t2t_train_chat.py` script.
