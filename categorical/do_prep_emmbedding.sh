#!/bin/bash

FROM=../raw/glove.6B.100d.txt
TO=../data/embed.txt


python3 -c "from gensim.scripts.glove2word2vec import glove2word2vec;glove2word2vec(glove_input_file='$FROM', word2vec_output_file='$TO')"

echo "file copied"