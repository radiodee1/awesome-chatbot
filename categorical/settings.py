#!/usr/bin/python3

hparams = {
    'save_dir': "../saved/",
    'data_dir': "../data/",
    'embed_name':'embed.txt',
    'vocab_name': "vocab.big",
    'test_name': "test",
    'test_size': 100,
    'train_name': "train",
    'src_ending': "from",
    'tgt_ending': "to",
    'base_filename': "chatbot-embedding",
    'base_file_num': 1,
    'num_train_total': 500000,
    'num_vocab_total': 2000,
    'batch_size': 256,#64, #256
    'steps_to_stats': 100,
    'embed_size':100,
    'infer_repeat': 2,
    'embed_mode':'normal', #values: mod, normal, zero
    'sol':'sol',
    'eol':'eol',
    'unk':'unk',
    'units': 128, #256 , #64,
    'learning_rate': 0.001, #0.001
    'tokens_per_sentence': 32,
    'raw_embedding_filename': 'embedding',
    'batch_constant': 512#384
    
}
