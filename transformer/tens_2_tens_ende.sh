# See what problems, models, and hyperparameter sets are available.
# You can easily swap between them (and add new ones).
t2t-trainer --registry_help

PROBLEM=translate_ende_wmt32k
MODEL=transformer
HPARAMS=transformer_base_single_gpu
PROJECT=./
#HOME=$PROJECT

DATA_DIR=$PROJECT/../saved/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$PROJECT/../saved/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

echo "************ here **************"


# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
#t2t-trainer --hparams='batch_size=1024' \
#  --data_dir=$DATA_DIR \
#  --problem=$PROBLEM \
#  --model=$MODEL \
#  --hparams_set=$HPARAMS \
#  --output_dir=$TRAIN_DIR

# Decode

DECODE_FILE=$DATA_DIR/decode_this.txt
echo "Hello world" >> ../raw/$DECODE_FILE
echo "Goodbye world" >> ../raw/$DECODE_FILE
echo -e 'Hallo Welt\nAuf Wiedersehen Welt' > ../saved/ref-translation.de

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=../raw/$DECODE_FILE \
  --decode_to_file=../saved/translation.en

# See the translations
cat ../saved/translation.en

# Evaluate the BLEU score
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
t2t-bleu --translation=../saved/translation.en --reference=../saved/ref-translation.de