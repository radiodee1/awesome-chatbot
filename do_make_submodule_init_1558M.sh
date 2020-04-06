cd model

./tf_gpt2_download_model.py 1558M

mkdir ../data/tf_gpt2_data/1558M/converted/

cd ../data/tf_gpt2_data/1558M/

transformers gpt2 model.ckpt converted/. ../../hp_config_1558.json

cp ../encoder.json converted/.
cp ../vocab.bpe converted/.

cp ../../hp_config_1558.json converted/config.json
