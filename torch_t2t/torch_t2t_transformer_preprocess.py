#!/usr/bin/python3

''' Handling the data io '''


'''
MIT License

Copyright (c) 2017 Victor Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import sys
sys.path.append('./t2t/')
sys.path.append('../')
import os
import argparse
from argparse import Namespace
import logging
import dill as pickle
import urllib
from tqdm import tqdm
import sys
import codecs
import spacy
import torch
import tarfile
import torchtext.data
import torchtext.datasets
from torchtext.datasets import TranslationDataset
import transformer.Constants as Constants
from learn_bpe import learn_bpe
from apply_bpe import BPE
from torchtext.data.utils import get_tokenizer

from model.tokenize_weak import format
from model.settings import hparams as hp

__author__ = "Yu-Hsiang Huang"



_TRAIN_DATA_SOURCES = [
    {"url": "http://data.statmt.org/wmt17/translation-task/" \
             "training-parallel-nc-v12.tgz",
     "trg": "news-commentary-v12.de-en.en",
     "src": "news-commentary-v12.de-en.de"},
    #{"url": "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
    # "trg": "commoncrawl.de-en.en",
    # "src": "commoncrawl.de-en.de"},
    #{"url": "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
    # "trg": "europarl-v7.de-en.en",
    # "src": "europarl-v7.de-en.de"}
    ]

_VAL_DATA_SOURCES = [
    {"url": "http://data.statmt.org/wmt17/translation-task/dev.tgz",
     "trg": "newstest2013.en",
     "src": "newstest2013.de"}]

_TEST_DATA_SOURCES = [
    {"url": "https://storage.googleapis.com/tf-perf-public/" \
                "official_transformer/test_data/newstest2014.tgz",
     "trg": "newstest2014.en",
     "src": "newstest2014.de"}]

def find_and_parse_story(data, period=False):
    print(len(data.examples), 'length')
    for ii in range(len(data.examples)):
        z = data.examples[ii]
        out = []
        for i in z.story:
            i = i.split(' ')
            for j in i:
                out.append(j)
            if period:
                out.append('.')
        #print(out)
        out.extend(z.query)
        out.append('?')
        data.examples[ii].story = out
        data.examples[ii].query.append('?')

        data.examples[ii].src = data.examples[ii].story[:] + [Constants.EOS_WORD]
        data.examples[ii].trg = data.examples[ii].answer[:] + [Constants.EOS_WORD]
        ## BOS_WORD added by model ##
        delattr(data.examples[ii], 'story')
        delattr(data.examples[ii], 'answer')
        delattr(data.examples[ii], 'query')

    return data

def find_and_parse_movie(name, max_len, start=0, mask=False):
    fr_name = '../data/' + name + '.big.from'
    to_name = '../data/' + name + '.big.to'
    data = Namespace()
    data.examples = []
    with open(fr_name,'r') as fr_file, open(to_name, 'r') as to_file:
        fr_list = fr_file.readlines()
        to_list = to_file.readlines()
        for ii in range(len(fr_list)):

            if ii >= max_len + start and max_len != -1: break
            if ii < start: continue

            #data.examples.append(None)
            #data.examples[ii - start] = Namespace(src='', trg='')
            fr_words = fr_list[ii]
            to_words = to_list[ii]
            fr_words = format(fr_words)
            to_words = format(to_words)

            fr_words = fr_words.replace(hp['eol'], '')
            fr_words = fr_words.replace(hp['sol'], '')

            to_words = to_words.replace(hp['eol'], '')
            to_words = to_words.replace(hp['sol'], '')

            fr_words = fr_words.split(' ')
            to_words = to_words.split(' ')

            if not mask:
                data.examples.append(None)
                data.examples[ii - start] = Namespace(src=list(), trg=list())
                data.examples[ii - start].src = fr_words + [Constants.EOS_WORD]
                data.examples[ii - start].trg = to_words + [Constants.EOS_WORD]
            else:
                skip = 2
                tokens = len(to_words)
                j = 0
                for jj in range(skip, tokens):
                    if len(to_words[skip:j]) == 0 :
                        j += 1
                        continue
                    data.examples.append(Namespace(src=list(), trg=list()))
                    data.examples[ii - start + j - skip - 1].src = fr_words[:] + [Constants.EOS_WORD]
                    data.examples[ii - start + j - skip - 1].trg.extend( to_words[:j])
                    j += 1
                    pass
                data.examples[ii - start + j - skip - 2].trg.append(Constants.EOS_WORD)
                #print(data.examples[ii - start + j - skip - 2].trg, 'trg', ii - start + j - skip - 2)

    return data


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def file_exist(dir_name, file_name):
    for sub_dir, _, files in os.walk(dir_name):
        if file_name in files:
            return os.path.join(sub_dir, file_name)
    return None


def download_and_extract(download_dir, url, src_filename, trg_filename):
    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)

    if src_path and trg_path:
        sys.stderr.write(f"Already downloaded and extracted {url}.\n")
        return src_path, trg_path

    compressed_file = _download_file(download_dir, url)

    sys.stderr.write(f"Extracting {compressed_file}.\n")
    with tarfile.open(compressed_file, "r:gz") as corpus_tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(corpus_tar, download_dir)

    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)

    if src_path and trg_path:
        return src_path, trg_path

    raise OSError(f"Download/extraction failed for url {url} to path {download_dir}")


def _download_file(download_dir, url):
    filename = url.split("/")[-1]
    if file_exist(download_dir, filename):
        sys.stderr.write(f"Already downloaded: {url} (at {filename}).\n")
    else:
        sys.stderr.write(f"Downloading from {url} to {filename}.\n")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
    return filename


def get_raw_files(raw_dir, sources):
    raw_files = { "src": [], "trg": [], }
    for d in sources:
        src_file, trg_file = download_and_extract(raw_dir, d["url"], d["src"], d["trg"])
        raw_files["src"].append(src_file)
        raw_files["trg"].append(trg_file)
    return raw_files


def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def compile_files(raw_dir, raw_files, prefix):
    src_fpath = os.path.join(raw_dir, f"raw-{prefix}.src")
    trg_fpath = os.path.join(raw_dir, f"raw-{prefix}.trg")

    if os.path.isfile(src_fpath) and os.path.isfile(trg_fpath):
        sys.stderr.write(f"Merged files found, skip the merging process.\n")
        return src_fpath, trg_fpath

    sys.stderr.write(f"Merge files into two files: {src_fpath} and {trg_fpath}.\n")

    with open(src_fpath, 'w') as src_outf, open(trg_fpath, 'w') as trg_outf:
        for src_inf, trg_inf in zip(raw_files['src'], raw_files['trg']):
            sys.stderr.write(f'  Input files: \n'\
                    f'    - SRC: {src_inf}, and\n' \
                    f'    - TRG: {trg_inf}.\n')
            with open(src_inf, newline='\n') as src_inf, open(trg_inf, newline='\n') as trg_inf:
                cntr = 0
                for i, line in enumerate(src_inf):
                    cntr += 1
                    src_outf.write(line.replace('\r', ' ').strip() + '\n')
                for j, line in enumerate(trg_inf):
                    cntr -= 1
                    trg_outf.write(line.replace('\r', ' ').strip() + '\n')
                assert cntr == 0, 'Number of lines in two files are inconsistent.'
    return src_fpath, trg_fpath


def encode_file(bpe, in_file, out_file):
    sys.stderr.write(f"Read raw content from {in_file} and \n"\
            f"Write encoded content to {out_file}\n")
    
    with codecs.open(in_file, encoding='utf-8') as in_f:
        with codecs.open(out_file, 'w', encoding='utf-8') as out_f:
            for line in in_f:
                out_f.write(bpe.process_line(line))


def encode_files(bpe, src_in_file, trg_in_file, data_dir, prefix):
    src_out_file = os.path.join(data_dir, f"{prefix}.src")
    trg_out_file = os.path.join(data_dir, f"{prefix}.trg")

    if os.path.isfile(src_out_file) and os.path.isfile(trg_out_file):
        sys.stderr.write(f"Encoded files found, skip the encoding process ...\n")

    encode_file(bpe, src_in_file, src_out_file)
    encode_file(bpe, trg_in_file, trg_out_file)
    return src_out_file, trg_out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-raw_dir', required=True)
    parser.add_argument('-data_dir', required=True)
    parser.add_argument('-codes', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-prefix', required=True)
    parser.add_argument('-max_len', type=int, default=100)
    parser.add_argument('--symbols', '-s', type=int, default=32000, help="Vocabulary size")
    parser.add_argument(
        '--min-frequency', type=int, default=6, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s))')
    parser.add_argument('--dict-input', action="store_true",
        help="If set, input file is interpreted as a dictionary where each line contains a word-count pair")
    parser.add_argument(
        '--separator', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument('--total-symbols', '-t', action="store_true")
    opt = parser.parse_args()

    # Create folder if needed.
    mkdir_if_needed(opt.raw_dir)
    mkdir_if_needed(opt.data_dir)

    # Download and extract raw data.
    raw_train = get_raw_files(opt.raw_dir, _TRAIN_DATA_SOURCES)
    raw_val = get_raw_files(opt.raw_dir, _VAL_DATA_SOURCES)
    raw_test = get_raw_files(opt.raw_dir, _TEST_DATA_SOURCES)

    # Merge files into one.
    train_src, train_trg = compile_files(opt.raw_dir, raw_train, opt.prefix + '-train')
    val_src, val_trg = compile_files(opt.raw_dir, raw_val, opt.prefix + '-val')
    test_src, test_trg = compile_files(opt.raw_dir, raw_test, opt.prefix + '-test')

    # Build up the code from training files if not exist
    opt.codes = os.path.join(opt.data_dir, opt.codes)
    if not os.path.isfile(opt.codes):
        sys.stderr.write(f"Collect codes from training data and save to {opt.codes}.\n")
        learn_bpe(raw_train['src'] + raw_train['trg'], opt.codes, opt.symbols, opt.min_frequency, True)
    sys.stderr.write(f"BPE codes prepared.\n")

    sys.stderr.write(f"Build up the tokenizer.\n")
    with codecs.open(opt.codes, encoding='utf-8') as codes: 
        bpe = BPE(codes, separator=opt.separator)

    sys.stderr.write(f"Encoding ...\n")
    encode_files(bpe, train_src, train_trg, opt.data_dir, opt.prefix + '-train')
    encode_files(bpe, val_src, val_trg, opt.data_dir, opt.prefix + '-val')
    encode_files(bpe, test_src, test_trg, opt.data_dir, opt.prefix + '-test')
    sys.stderr.write(f"Done.\n")


    field = torchtext.data.Field(
        tokenize=str.split,
        lower=True,
        pad_token=Constants.PAD_WORD,
        init_token=Constants.BOS_WORD,
        eos_token=Constants.EOS_WORD)

    fields = (field, field)

    MAX_LEN = opt.max_len

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    enc_train_files_prefix = opt.prefix + '-train'
    train = TranslationDataset(
        fields=fields,
        path=os.path.join(opt.data_dir, enc_train_files_prefix),
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    from itertools import chain
    field.build_vocab(chain(train.src, train.trg), min_freq=2)

    data = { 'settings': opt, 'vocab': field, }
    opt.save_data = os.path.join(opt.data_dir, opt.save_data)

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    pickle.dump(data, open(opt.save_data, 'wb'))



def main_wo_bpe():
    '''
    Usage: python preprocess.py -lang_src de -lang_trg en -save_data multi30k_de_en.pkl -share_vocab
    '''

    spacy_support_langs = ['de', 'el', 'en', 'es', 'fr', 'it', 'lt', 'nb', 'nl', 'pt']

    parser = argparse.ArgumentParser()
    parser.add_argument('-lang_src', required=False, choices=spacy_support_langs, default='en')
    parser.add_argument('-lang_trg', required=False, choices=spacy_support_langs, default='en')
    parser.add_argument('-save_data', required=False, default='../data/data_transformer.bin')
    parser.add_argument('-data_src', type=str, default=None)
    parser.add_argument('-data_trg', type=str, default=None)

    parser.add_argument('-max_len', type=int, default=-1)
    parser.add_argument('-min_word_count', type=int, default=1)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true', default=True)
    #parser.add_argument('-ratio', '--train_valid_test_ratio', type=int, nargs=3, metavar=(8,1,1))
    #parser.add_argument('-vocab', default=None)
    parser.add_argument('-tenk', action='store_true', help='use ten-k dataset')
    parser.add_argument('-task', default=1, help='use specific question-set/task', type=int)
    parser.add_argument('-movie', action='store_true', help='use movie corpus.')
    parser.add_argument('-movie_start', help='skip movie corpus lines.', default=0, type=int)
    parser.add_argument('-vocab_file', help='path to separate vocab file.',default='../data/data_vocab.bin')
    parser.add_argument('-movie_stagger', action='store_true', help='enable movie target staggering.')

    opt = parser.parse_args()
    assert not any([opt.data_src, opt.data_trg]), 'Custom data input is not support now.'
    assert not any([opt.data_src, opt.data_trg]) or all([opt.data_src, opt.data_trg])
    print(opt)

    if opt.movie:
        opt.save_data = '../data/data_transformer.bin'

    src_lang_model = spacy.load(opt.lang_src)
    trg_lang_model = spacy.load(opt.lang_trg)

    def tokenize_src(text):
        return [tok.text for tok in src_lang_model.tokenizer(text)]

    def tokenize_trg(text):
        return [tok.text for tok in trg_lang_model.tokenizer(text)]

    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token=Constants.BOS_WORD, #'<sos>',
                                eos_token=Constants.EOS_WORD, #'<eos>',
                                lower=True)


    MAX_LEN = opt.max_len
    MIN_FREQ = 0 #opt.min_word_count
    MOVIE_START = opt.movie_start
    MOVIE_ANY_LEN = -1
    MOVIE_ANY_START = 0

    if MAX_LEN < 500:
        MOVIE_ANY_LEN = MAX_LEN
        MOVIE_ANY_START = 0

    if not all([opt.data_src, opt.data_trg]):
        assert {opt.lang_src, opt.lang_trg} == {'en', 'en'}
    else:
        # Pack custom txt file into example datasets
        raise NotImplementedError

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN


    if not opt.movie:
        train, val, test = torchtext.datasets.BABI20.splits(
            TEXT,
            root='../raw/',
            tenK=opt.tenk,
            task=opt.task,
        )

        train = find_and_parse_story(train, period=True)
        val = find_and_parse_story(val, period=True)
        test = find_and_parse_story(test, period=True)

    if opt.movie:
        train = find_and_parse_movie('train', MAX_LEN, MOVIE_START, mask=opt.movie_stagger)
        val = find_and_parse_movie('valid', MOVIE_ANY_LEN, MOVIE_ANY_START, mask=opt.movie_stagger)
        test = find_and_parse_movie('test', MOVIE_ANY_LEN, MOVIE_ANY_START)
        pass

    ## print some values
    print(list(i.src for i in train.examples[:3]), '< src')

    vocab = []
    for i in train.examples[:]:
        vocab.extend([i.src[:]])


    TEXT.build_vocab(vocab, min_freq=MIN_FREQ)
    print('[Info] Get text language vocabulary size:', len(TEXT.vocab))

    '''
    if opt.share_vocab and False:
        print('[Info] Merging two vocabulary ...')
        for w, _ in SRC.vocab.stoi.items():
            # TODO: Also update the `freq`, although it is not likely to be used.
            if w not in TRG.vocab.stoi:
                TRG.vocab.stoi[w] = len(TRG.vocab.stoi)
                TEXT.vocab.stoi[w] = len(TRG.vocab.stoi)
        TRG.vocab.itos = [None] * len(TRG.vocab.stoi)
        TEXT.vocab.itos = [None] * len(TRG.vocab.stoi)
        for w, i in TRG.vocab.stoi.items():
            TRG.vocab.itos[i] = w
            TEXT.vocab.itos[i] = w
        SRC.vocab.stoi = TRG.vocab.stoi
        SRC.vocab.itos = TRG.vocab.itos

        print('[Info] Get merged vocabulary size:',  len(TEXT.vocab))
    '''
    #print(TEXT.vocab.stoi)
    SRC = TRG = TEXT

    data = {
        'settings': opt,
        'vocab': {'src': SRC, 'trg': TRG, 'txt': TEXT},
        'train': train.examples,
        'valid': val.examples,
        'test': test.examples}

    voc_data = {
        'vocab': {'src': SRC, 'trg': TRG, 'txt': TEXT}
    }

    print(data['train'][0].src, data['train'][0].trg)
    print(len(data['train']), 'train' ,len(data['valid']), 'valid', len(data['test']), 'test')

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    pickle.dump(data, open(opt.save_data, 'wb'))

    if not os.path.isfile(opt.vocab_file):
        pickle.dump(voc_data, open(opt.vocab_file, 'wb'))
        print(opt.vocab_file, 'NEW VOCAB FILE')
    else:
        print(opt.vocab_file, 'file exists. NOT REPLACING.')

if __name__ == '__main__':
    main_wo_bpe()
    #main()
