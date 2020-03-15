cd model

./tf_gpt2_download_model.py 117M

mkdir -p ../data/tf_gpt2_data/117M/converted/

cd ../data/tf_gpt2_data/117M/

transformers gpt2 ../model.ckpt converted/.  #../../hp_config.json

cp ../encoder.json converted/.
cp ../vocab.bpe converted/.

#cp ../config.json converted/config.json
