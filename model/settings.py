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
    'question_ending':'ques',
    'babi_name':'babi',
    'babi_memory_hops': 5,
    'base_filename': "weight",
    'base_file_num': 1,
    'stats_filename':'stat',
    'num_vocab_total': 15000,
    'batch_size': 256,#64, #256
    'steps_to_stats': 50,
    'epochs': 500,
    'embed_size':None, #values only: 50, 100, 200, 300, None for none
    'pytorch_embed_size': 100,
    'embed_train':False,
    'autoencode':0.0,
    'infer_repeat': 1,
    'embed_mode':'normal', #values only: mod, normal, zero
    'dense_activation':'tanh', #values only: tanh, relu, softmax, none
    'sol':'sol',
    'eol':'eol',
    'unk':'unk',
    'units': 100, #128, #256 , #64,
    'layers':2,
    'teacher_forcing_ratio': 0.0, ## 0.5
    'dropout': 0.2,
    'learning_rate': 0.001, #0.0001
    'tokens_per_sentence': 30, #32,
    'batch_constant': 512, #
    'zero_start': False
}
