#!/usr/bin/python3.6


import os
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer

os.mkdir("./tmp/")
os.environ['ZEROMQ_SOCK_TMP_DIR'] = "./tmp/"

args_bert = [
        "-model_dir=../data/bert_data/uncased_L-12_H-768_A-12/" ,
        "-num_worker=4" ,
        '-max_batch_size=1', ## speed considerations !!
        "-pooling_strategy=LAST_TOKEN",
        "-cpu",
        "-pooling_layer=-2",
        "-max_seq_len=15",
        "-show_tokens_to_client",
        #"-output_dir="
    ]

if __name__ == '__main__':
    args = get_args_parser().parse_args(args_bert)

    server = BertServer(args)
    server.start()
    '''
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.argv.extend(args_bert)
    sys.exit(main())
    '''