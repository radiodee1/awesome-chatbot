# Start Training Code: `babi.py`

It should be noted that the code in this repository uses a batch size of 1, essentially no batch size. Also gpu processing is not implemented. This slows down training.

#### Setup for `small` and `small_tr`:
1. move to the root directory of the repository.
2. `./do_make_glove_download.sh` Execute this in the root folder of the project repository.
3. `cd model`
4. `./do_make_train_test_from_babi.py all en`
5. `./do_make_vocab.py --babi --load-embed-size=100`
6. `./do_make_train_test_from_babi.py 1 en-10k` NOTE: replace '1' with current test.
7. `cd ..`
8. `./do_launch_babi.sh --mode=long --units=100 --load-babi --load-embed-size=100 --hide-unk --basename=small --freeze-embedding --lr=0.001 --babi-num=1` NOTE: replace '1' with current test. Remove '--freeze-embedding' and '--load-embed-size=100' as necessary for test.

#### Setup for `lrg`:
Here we are interested in creating the same conditions as above but with a larger vocabulary file. Our goal vocabulary size is greater than 1500 words.

1. move to the root directory of the repository.
2. `./do_make_glove_download.sh`
3. `cd model`
4. `./do_make_train_test_from_babi.py all en`
5. `./do_make_vocab.py --babi --all-glove --load-embed-size=100`
6. `./do_make_train_test_from_babi.py 1 en-10k` NOTE: replace '1' with current test.
7. `cd ..`
8. `./do_launch_babi.sh --mode=long --units=100 --load-babi --load-embed-size=100 --hide-unk --basename=lrg --freeze-embedding --lr=0.001 --babi-num=1` NOTE: replace '1' with current test. Remove '--freeze-embedding' and '--load-embed-size=100' as necessary for test.
