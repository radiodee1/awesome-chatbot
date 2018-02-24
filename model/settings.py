#!/usr/bin/python3

hparams = {
    'save_dir': "../saved/",
    'data_dir': "../data/",
    'vocab_name': "vocab",
    'test_name': "test",
    'test_size': 100,
    'train_name': "train",
    'src_ending': "from",
    'tgt_ending': "to",
    'base_filename': "chatbot",
    'base_file_num': 1,
    'num_train_total': 500000,
    'num_vocab_total': 20000,
    'batch_size': 16, #256
    'steps_to_stats': 100,
    'sol':'sol',
    'eol':'eol',
    'unk':'unk',
    'layers': 2,
    'units': 50, ##600
    'learning_rate': 0.001,
    'tokens_per_sentence': 55,
    'raw_embedding_filename': 'embedding'
    
}
