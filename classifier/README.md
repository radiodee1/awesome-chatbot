## classifier

These files run a simple classifier.

* `do_make_bert_interactive.sh` - run this file for interactive use of a trained
bert model.
* `do_make_bert_train.sh` - run this file for training a bert model.
* `run_classifier.py` - this file is run by the scripts above. Do not run from the command line.

Use this format:

```
INFILE=t2t .do_make_bert_train.sh
```
This will look for a tab file with the `t2t` signature for training. Use the same
format for calling the interactive shell script as follows.

```
INFILE=t2t .do_make_bert_interactive.sh
```

The versions available are:
* `INFILE=t2t`
* `INFILE=gpt`
* `INFILE=gru`
