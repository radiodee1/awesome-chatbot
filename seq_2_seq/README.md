### Files

* `do_make_train_test_from_db.py` - runnable file to make train and test files found in `../data/`
* `do_make_vocab.py` - runnable file for creating vocab file found in `../data/`
* `do_split.py` - runnable file for processing train and test files found in `../data/`
* `README.md` - this file.
* `seq_2_seq.py` - non-working hand coded example.
* `seq_2_seq_tutorial.py` - working Inkawhich tutorial file.
* `settings.py` - non-runnable hyper-parameter file.
* `tokenize_weak.py` - non runnable tokenizing function.

## Run `seq_2_seq` Model

* This code runs but produces no usable output.

```
./seq_2_seq.py --mode long --basename somename_01 --hide-unk --skip-unk --lr 0.002 --length 25 --load-recurrent --units 500 --single
```

* This code runs and uses the Inkawhich tutorial. It does produce usable output.

```
./seq_2_seq_tutorial.py --mode long --build-train-data --train
```

* This line uses the 'tutorial' code in inference mode.
```
./seq_2_seq_tutorial.py --mode interactive --iter 4000
```