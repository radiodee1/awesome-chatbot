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
8. `./do_launch_babi.sh --mode=long --units=100 --load-babi --load-embed-size=100 --hide-unk --basename=small --freeze-embedding --lr=0.001 --babi-num=1` NOTE: replace '1' with current test. Replace '--lr=0.001' with '--lr=0.00001'. Remove '--freeze-embedding' as necessary for test. Change '--basename=ii_a' as necessary.

#### Setup for `lrg`:
Here we are interested in creating the same conditions as above but with a larger vocabulary file. To do this we download the reddit corpus and use it to make large 'train.big.from' and 'train.big.to' files. This process is long and laborious. Our goal vocabulary size is 1500 words. To achieve this we could also download the wiki corpus. The name of the resulting file is `vocab.big.txt`.

1. move to the root directory of the repository.
2. `./do_make_glove_download.sh`
3. `cd model`
4. `./do_make_train_test_from_babi.py all en`
5. `./do_make_vocab.py --babi --all-glove --load-embed-size=100`
6. `./do_make_train_test_from_babi.py 1 en-10k` NOTE: replace '1' with current test.
7. `cd ..`
8. `./do_launch_babi.sh --mode=long --units=100 --load-babi --load-embed-size=100 --hide-unk --basename=lrg --freeze-embedding --lr=0.001 --babi-num=1` NOTE: replace '1' with current test. Replace '--lr=0.001' with '--lr=0.00001'. Remove '--freeze-embedding' as necessary for test. Change '--basename=ii_a' as necessary.
