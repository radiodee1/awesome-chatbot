# Start Training Code: `babi_ii.py`

It should be noted that the code in this repository uses a batch size of 1, essentially no batch size. This slows down training.

#### Setup for `small_e` and `small_tr`:
1. `./do_make_glove_download.sh` Execute this in the root folder of the project repository.
2. `cd model`
3. `./do_make_train_test_from_babi.py all`
4. `./do_make_vocab.py ../data/train.babi.from --babi --babi-only --load-embed-size=100`
5. `./do_make_train_test_from_babi.py 1` NOTE: replace '1' with current test.
6. `cd ..`
7. `./do_launch_babi.sh --mode=long --units=100 --load-babi --load-embed-size=100 --hide-unk --basename=small_e --freeze-embedding --lr=0.001 --babi-num=1` NOTE: replace '1' with current test. Replace '--lr=0.001' with '--lr=0.00001'. Remove '--freeze-embedding' as necessary for test. Change '--basename=ii_a' as necessary.
8. `cd stats`
9. `./stats.py --test --mode=long --units=100 --load-babi --load-embed-size=100 --hide-unk --basename=small_e --freeze-embedding --lr=0.001 --babi-num=1` NOTE: replace '1' with current test. Replace '--lr=0.001' with '--lr=0.00001'. Remove '--freeze-embedding' as necessary for test. *THIS WILL PLACE A TESTING VALUE IN THE 'STAT.MD' CHART.*

#### Setup for `ii_a`, `ii_b` and `ii_tr`:
Here we are interested in creating the same conditions as above but with a larger vocabulary file. To do this we download the reddit corpus and use it to make large 'train.big.from' and 'train.big.to' files. This process is long and laborious. Our goal vocabulary size is 1500 words. To achieve this we could also download the wiki corpus. The name of the resulting file is `vocab.big.txt`.

1. move to the root directory of the repository.
2. `./do_make_glove_download.sh`
3. `./do_make_reddit_download.sh` Note: this can take hours
4. `cd raw`
5. `bunzip2 RC_2015-01.bz2` Note: this can take hours and consumes between 20 and 50 Gigabytes.
6. `cd ..`
7. `./do_make_db_from_reddit.py` Note: this can take hours. You can let it run or stop it when you have the desired number of paired rows. We are looking for a number like 500,000 paired rows. Ctrl-c will stop the program.
8. `cd model`
9. `./do_make_train_test_from_db.py ../input.db` Note: this can take hours.
10. `./do_make_vocab.py ../data/train.big.from --babi --load-embed-size=100`
11. Start at step number 5 above.
12. `./do_make_train_test_from_babi.py 1` NOTE: replace '1' with current test.
13. `cd ..`
14. `./do_launch_babi.sh --mode=long --units=100 --load-babi --load-embed-size=100 --hide-unk --basename=ii_a --freeze-embedding --lr=0.001 --babi-num=1` NOTE: replace '1' with current test. Replace '--lr=0.001' with '--lr=0.00001'. Remove '--freeze-embedding' as necessary for test. Change '--basename=ii_a' as necessary.
15. `cd stats`
16. `./stats.py --test --mode=long --units=100 --load-babi --load-embed-size=100 --hide-unk --basename=ii_a --freeze-embedding --lr=0.001 --babi-num=1` NOTE: replace '1' with current test. Replace '--lr=0.001' with '--lr=0.00001'. Remove '--freeze-embedding' as necessary for test. *THIS WILL PLACE A TESTING VALUE IN THE 'STAT.MD' CHART.*
