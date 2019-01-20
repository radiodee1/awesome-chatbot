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

#### Recurrent Output: `babi_recurrent.py`:
Test 19 uses two word output. The babi.py program that I've been working on is not set up to reply with more than one word at a time.
To address this a gru was added to the output of the model. Test 19 is then supposed to be run with the new code. The code allows for
command line options that set the number of layers in the recurrent module. Layer size of 1 and 2 was tried and 1 was found to be best.
Even with one recurrent layer testing output was never very good. See the table below.

| Test | 1 recurrent layer | 2 recurrent layers |
|-|-|-|
| Test 19 | 7.39% | 30% |

At the same time as test 19 is being run with the recurrent code the model is being experimented with on a different computer with a different set of goals.
Here we are interested in finding a way to make a chatbot using the babi_recurrent.py code.

The major obstacle that I see for this right now is that I don't have a good data set at my disposal. I currently have access to a movie data base and a reddit dump data base.
Both are large but noise filled. I find that training with question/answer pairs is not very predictable. If it fails totally we should have another option.

What we could do, and should try, is auto-encode the recurrent module with the movie corpus. Then we'll freeze the output module.
Then we'll train the rest of the model over again with question/answer oriented data. This would be all with the DMN type architecture that we've been working with to solve the babi tasks.

We could also employ a seq2seq architecture. This might be easy to implement. The topic is discussed in the following paper. [See Here.](http://arxiv.org/abs/1506.05869v3) This model uses one input and one output. The babi model uses two inputs and one output. Both should be tried.

Then maybe we'll compare our results using the DMN type model with a seq2seq chatbot architecture.

### Part Of Speech: `babi_recurrent.py`:
We downloaded a POS dataset from kaggle. The babi_recurrent.py code was modified slightly and
used for this test. Our scores were in the mid 90% range.