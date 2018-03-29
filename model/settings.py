#!/usr/bin/python3

hparams = {
    'save_dir': "../saved/",
    'data_dir': "../data/",
    'embed_name':'embed.txt', #used for glove. Note: glove vectors don't have contractions!!
    'vocab_name': "vocab.big.txt", ## if you set this to None the pytorch model should work with 2 languages
    'test_name': "test",
    'test_size': 100,
    'train_name': "train",
    'src_ending': "from",
    'tgt_ending': "to",
    'base_filename': "bidirectional-add",
    'base_file_num': 1,
    'num_vocab_total': 20000,
    'batch_size': 256,#64, #256
    'steps_to_stats': 50,
    'epochs': 100,
    'embed_size':None, #values only: 50, 100, 200, 300, None for none
    'embed_train':False,
    'autoencode':False,
    'infer_repeat': 1,
    'embed_mode':'normal', #values only: mod, normal, zero
    'dense_activation':'tanh', #values only: tanh, relu, softmax, none
    'sol':'sol',
    'eol':'eol',
    'unk':'unk',
    'units': 512, #128, #256 , #64,
    'learning_rate': 0.001, #0.001
    'tokens_per_sentence': 18, #32,
    'batch_constant': 512 #
    
}
