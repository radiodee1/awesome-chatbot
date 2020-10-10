## Running `chat_mod_calc.py`:

* `cd` to the root of the project directory.
* Set some environment varibales. The two major vars are `STAT_LIMIT` and `ORIGINAL_SENTENCES`. `STAT_LIMIT` is the number of sentences to process. `ORIGINAL_SENTENCES` is `1` or `0`.
* Call one of the scripts that starts with `do_calc_` and ends with a model type (`gpt2`, `gru`, or `t2t`) and the extension `.sh`.
* Redirect a set of sentence inputs into the runnable shell script. An example would be `data/train.big.from`.
* Output will be in the `saved` directory. Output will be in the form of `output.original_xxx.###.csv`. Here `xxx` is the model type from above. `###` should match the `STAT_LIMIT` environment variable.

Two examples follow:
```
$ cd ..
$ ORIGINAL_SENTENCES=1 STAT_LIMIT=100000 ./do_calc_gru.sh data/train.big.from
$ ORIGINAL_SENTENCES=1 STAT_LIMIT=100000 ./do_calc_t2t.sh data/train.big.from
```

In the examples above the `calc` script goes through 100,000 sentences. In the first example the `gru` model is tested. This run takes less than an hour.

In the second example the `calc` script runs the `t2t` model. This takes several days.

The `gpt2` model takes as long as the `t2t`. The `raw` model processes the data without any neural network at all and is very fast.

The output files have data arranged in rows, not columns. To graph these `csv` files execute `libreoffice output.original_t2t.100000.csv`. This opens a small window that askes how to load the csv text file. Then it opens a `libreoffice` spreadsheet.

