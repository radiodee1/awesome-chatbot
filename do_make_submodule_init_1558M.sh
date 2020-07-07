cd model


./tf_gpt2_download_model.py 1558M

mkdir ../data/tf_gpt2_data/1558M/converted/

cd ../data/tf_gpt2_data/1558M/

## this works with transformers 2.0.0

#transformers gpt2 model.ckpt converted/. ../../hp_config_1558.json

transformers-cli convert --model_type gpt2 \
  --tf_checkpoint model.ckpt \
  --pytorch_dump_output converted/. \
  --config ../../hp_config_1558.json

cp ../encoder.json converted/.
cp ../vocab.bpe converted/.

cp ../../hp_config_1558.json converted/config.json
