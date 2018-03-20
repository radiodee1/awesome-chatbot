#!/usr/bin/python3

hparams = {
    'save_dir': "../saved/",
    'data_dir': "../data/",
    'embed_name':'embed.txt', #used for glove
    'vocab_name': "vocab.big",
    'test_name': "test",
    'test_size': 100,
    'train_name': "train",
    'src_ending': "from",
    'tgt_ending': "to",
    'base_filename': "chatbot-att-no-shift",
    'base_file_num': 1,
    'num_train_total': 500000, #replaced by epochs
    'num_vocab_total': 2000,
    'batch_size': 256,#64, #256
    'steps_to_stats': 100,
    'epochs': 100,
    'embed_size':100, #values only: 50, 100, 200, 300
    'embed_train':True,
    'autoencode':False,
    'infer_repeat': 1,
    'embed_mode':'normal', #values only: mod, normal, zero
    'dense_activation':'tanh',#'tanh', #values only: tanh, relu, softmax, none
    'sol':'sol',
    'eol':'eol',
    'unk':'unk',
    'units': 128, #256 , #64,
    'learning_rate': 0.001, #0.001
    'tokens_per_sentence': 32,
    'raw_embedding_filename': 'embedding', #not used with glove
    'batch_constant': 512 #
    
}
