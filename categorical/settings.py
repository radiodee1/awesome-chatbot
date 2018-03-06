#!/usr/bin/python3

hparams = {
    'save_dir': "../saved/",
    'data_dir': "../data/",
    'vocab_name': "vocab.big",
    'test_name': "test",
    'test_size': 100,
    'train_name': "train",
    'src_ending': "from",
    'tgt_ending': "to",
    'base_filename': "chatbot-categorical",
    'base_file_num': 1,
    'num_train_total': 500000,
    'num_vocab_total': 2000,
    'batch_size': 64, #256
    #'steps_to_stats': 100,
    'sol':'sol',
    'eol':'eol',
    'unk':'unk',
    #'layers': 2,
    'units': 64, ##600
    'learning_rate': 0.001,
    'tokens_per_sentence': 25,
    'raw_embedding_filename': 'embedding',
    'batch_constant': 1600
    
}
