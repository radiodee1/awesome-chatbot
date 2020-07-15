#!/usr/bin/env python3

hparams = {
    'save_dir': "../saved/",
    'data_dir': "../data/",
    'embed_name':'embed.txt', #used for glove. Note: glove vectors don't have contractions!!
    'vocab_name': "vocab.big.txt", ## if you set this to None the pytorch model should work with 2 languages
    'test_name': "test",
    'test_size': 100,
    'train_name': "train",
    'valid_name':'valid',
    'src_ending': "from",
    'tgt_ending': "to",
    'question_ending':'ques',
    'hist_ending': 'hist',
    'babi_name':'babi',
    'pos_name': 'pos',
    'babi_memory_hops': 4,
    'base_filename': "test_weights",
    'base_file_num': 1,
    'stats_filename':'stat',
    'num_vocab_total': 5000,
    'batch_size': 50,#64, #256
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
    'eow':'eow',  # end of word
    'units': 100, #128, #256 , #64,
    'layers':2,
    'decoder_layers': 2,
    'teacher_forcing_ratio': 0.0 , #1.0, ## 0.5
    'dropout': 0.0,
    'learning_rate': 0.01, # adam = 0.001, adagrad = 0.01
    'weight_decay': 0, #5e-4,
    'tokens_per_sentence': 10, #10,
    'batch_constant': 512, #
    'zero_start': False,
    'cuda': False,
    'split_sentences': True,
    'multiplier': 1.0, #0.5
    'beam': None,
    'single': False
}
