#!/bin/bash

echo "you want to put your data files in the new_data/ folder"
echo "then run this command again."
echo "this sets up a pair of vocabulary files. You don't want to"
echo "change them after you start training... even if you change"
echo "your train.from and train.to files."

cd setup
python3 prepare_data.py 20000
cd ..
cp new_data/t* data/.

cd data/
echo "change names here?"
#ls -hal
mv vocab.big.from vocab.from
mv vocab.big.to vocab.to