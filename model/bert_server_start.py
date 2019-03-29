#!/usr/bin/python3.6


import re
import sys

from bert_serving.server.cli import main

args_bert = [
        "-model_dir=../data/bert_data/uncased_L-12_H-768_A-12/" ,
        "-num_worker=4" ,
        '-max_batch_size=1', ## speed considerations !!
    ]

if __name__ == '__main__':

    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.argv.extend(args_bert)
    sys.exit(main())
