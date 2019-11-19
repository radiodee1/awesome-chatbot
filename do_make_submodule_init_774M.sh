cd model

./tf_gpt2_download_model.py 774M

mkdir ../data/tf_gpt2_data/774M/converted/

cd ../data/tf_gpt2_data/774M/

transformers gpt2 model.ckpt converted/. ../../hp_config.json

cp ../encoder.json converted/.
cp ../vocab.bpe converted/.

cp ../../hp_config.json converted/config.json
