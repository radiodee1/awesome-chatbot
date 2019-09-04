## Experiments

Presently the experiment is to try to complete the bAbI tests
on the GPT2 model architecture.

* `tf_gpt2_train_babi.py` - This file trains and tests the gpt2 model
using the bAbI dataset and the tensorflow backend.

* `do_make_train_test_from_babi.sh` - Run this file with certain parameters for setting up which bAbI 
experiment you want to run. This file actually runs a python script in
the `../model/` folder. The parameters are 'num' which is a number, and the phrase
'en-10k' to select the 10k example models.

* `torch_gpt2_load_babi.py`- This file runs the pytorch saved gpt2 file and
loads the bAbI test set.

* `../model/tf_gpt2_torch_convert.py` - This file converts the tensorflow model to the pytorch representation. 
This conversion goes from an arbitrary tensorflow checkpoint to
a single pytorch file saved in an arbitrary folder. You generally cannot
pick the output file's name.

* `../do_make_rename_train_valid_test.sh` - Change to this directory and run this commmand with the single parameter 'babi'. This will
make sure that the babi test set is the one that the symbolic
links in the 'data' folder points to.

First you train and test the tensorflow model. To do this you must first
set up the babi training files. Then you can
convert the model to the pytorch model, and then for confirmation
you can run the test set on the pytorch model to see if it
matches the test results on the tensorflow background.

## Transformer
There is also a file in this folder named `tf_t2t_train_babi.py`. This file
runs the tensor2tensor code that works with the babi set. The file
just runs the executable that ends up on the system when you install the google
tensor2tensor repository using `pip3`. 