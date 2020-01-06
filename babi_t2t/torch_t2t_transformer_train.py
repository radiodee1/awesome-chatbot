#!/usr/bin/python3
'''
This script handles the training process.
'''

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
import argparse
import math
import time
import dill as pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
import torchtext
from model.settings import hparams

__author__ = "Yu-Hsiang Huang"

babi_train_txt_in, babi_val_txt_in, babi_test_txt_in = None, None, None
babi_train_txt, babi_val_txt, babi_test_txt = None, None, None

TEXT = None

def load_q_a(ten_k, task):
    global babi_train_txt_in, babi_val_txt_in, babi_test_txt_in
    global TEXT
    from torchtext.data.utils import get_tokenizer
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)

    babi20 = torchtext.datasets.BABI20
    babi_train_txt_in, babi_val_txt_in, babi_test_txt_in = babi20.splits(TEXT, root='../raw/',tenK=ten_k, task=task)

    pass

def find_and_parse_story(data, period=False, iters=False):
    if iters: return data
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
        data.examples[ii].story = out
        data.examples[ii].query.append('?')
    return data

def batchify_babi(data, bsz, separate_ques=True, size_src=200, size_tgt=200, print_to_screen=False, device='cpu'):
    device = torch.device(device)
    new_data = []
    target_data = []
    for ii in range(len(data.examples)):
        z = data.examples[ii]
        target_data_tmp = [] #['<sos>']
        if not separate_ques:
            z.story.extend(z.query)
            z.story.extend('.')
            new_data.extend(z.story)
            new_data.append('<eos>')
            target_data_tmp.extend(z.story)
            target_data_tmp.extend(z.answer)
            target_data_tmp.append('<eos>')
            #print(z.answer, len(z.answer))
            ll = 2
            target_data_tmp = target_data_tmp[ll :len(z.story) + ll]
            #print(z.story,'\n',target_data_tmp)
            target_data.extend(target_data_tmp)
        else:
            z.story.insert(0, '<sos>')
            z.story.extend(z.query)
            z.story.extend([ '<eos>'])
            #z.story.extend('.')
            new_data.append(z.story)
            target_data_tmp.extend(z.answer)
            target_data_tmp.append('<eos>')
            target_data.append(target_data_tmp)

        pass
    if print_to_screen: print(new_data[0:5],'nd')
    m = max([len(x) for x in new_data])
    n = max([len(x[0]) for x in target_data])
    m = max(m, size_src)
    #n = max(n, size_tgt)
    n = m
    #print(m,'m', [len(x) for x in new_data])
    if not separate_ques:

        new_data = TEXT.numericalize([new_data])
        target_data = TEXT.numericalize([target_data])

        #new_n_data = new_data
        #target_n_data = target_data
        bsz = n
        nbatch_s = new_data.size(0) // bsz
        nbatch_t = target_data.size(0) // bsz
        nbatch = min(nbatch_s, nbatch_t)
        #print(nbatch_s, nbatch_t, len(new_data), len(target_data))
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        new_data = new_data.narrow(0, 0, nbatch * bsz)
        target_data = target_data.narrow(0, 0, nbatch * bsz)
        ###target_data = target_data.narrow(0, 0, nbatch * bsz)
        #print(new_data.size(), target_data.size())

        # Evenly divide the data across the bsz batches.
        new_n_data = new_data.view(bsz, -1).t().contiguous()
        target_n_data = target_data.view(bsz, -1).t().contiguous()

    else:

        #padded_data = torch.zeros(1, m, dtype=torch.long)
        #padded_target = torch.zeros(1, n, dtype=torch.long)
        new_n_data = torch.zeros( len(new_data), m, dtype=torch.long)
        target_n_data = torch.zeros( len(target_data), n, dtype=torch.long)

        for jj in range(len(new_data)):
            ## do source ##
            z = TEXT.numericalize([new_data[jj]])
            if z.size(0) > 1:
                z = z.t()
            p = torch.zeros(1, m, dtype=torch.long)
            p[0, :len(z[0])] = z
            new_n_data[jj, :] = p
            ## do target ##
            y = TEXT.numericalize([target_data[jj]])
            if y.size(0) > 1:
                y = y.t()
            q = torch.zeros(1, n, dtype=torch.long)
            q[0,:len(y[0])] = y
            target_n_data[jj, :] = q

        new_n_data = new_n_data.t().contiguous()
        #new_n_data = new_n_data.contiguous()
        target_n_data = target_n_data.t().contiguous()
        #print(new_n_data.t()[0:5], 't-nnd')

    return new_n_data.to(device), target_n_data.to(device), m

def get_batch_babi(source, target, i, print_to_screen=False, bptt=35, flatten_target=True):
    seq_len = bptt
    data = source[:, i : i +  seq_len]
    target = target[:, i : i  + seq_len]
    #print(label, bptt, i, 'lbl', data.size())
    if flatten_target:
        target = target.view(-1)
    if print_to_screen: print(data, target, i, 'dti')
    return data, target

def show_strings(source):
    if len(source.size()) > 1:
        source = source.squeeze(0)
    for i in source:
        if i != 0:
            print(TEXT.vocab.itos[i], end=' | ')
    print()

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0 

    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        # prepare data
        src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)

        # backward and update parameters
        loss, n_correct, n_word = cal_performance(
            pred, gold, opt.trg_pad_idx, smoothing=smoothing) 
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

            # forward
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(
                pred, gold, opt.trg_pad_idx, smoothing=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file, log_valid_file = None, None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, loss, accu, start_time):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=math.exp(min(loss, 100)),
                  accu=100*accu, elapse=(time.time()-start_time)/60))

    #valid_accus = []
    valid_losses = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, opt, device, smoothing=opt.label_smoothing)
        print_performances('Training', train_loss, train_accu, start)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt)
        print_performances('Validation', valid_loss, valid_accu, start)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main():
    ''' 
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 256 -warmup 128000
    '''
    global TEXT
    global babi_train_txt_in, babi_val_txt_in, babi_test_txt_in
    global babi_train_txt, babi_val_txt, babi_test_txt

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default=None)     # all-in-1 data pickle or bpe field

    parser.add_argument('-train_path', default=None)   # bpe encoded data
    parser.add_argument('-val_path', default=None)     # bpe encoded data

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true', default=True)
    parser.add_argument('-proj_share_weight', action='store_true', default=True)

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default='../saved/t2t_model.tar')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true', default=True)
    parser.add_argument('-label_smoothing', action='store_true')

    parser.add_argument('--tenk', action='store_true', help='use ten-k dataset')
    parser.add_argument('--task', default=1, help='use specific question-set/task', type=int)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    if not opt.log and not opt.save_model:
        print('No experiment result will be saved.')
        exit()
        raise

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n'\
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'\
              'Using smaller batch w/o longer warmup may cause '\
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if opt.cuda else 'cpu')
    print(device,'device')
    
    #========= Loading Dataset =========#
    load_q_a(opt.tenk, opt.task)

    babi_train_txt = find_and_parse_story(babi_train_txt_in, period=True)
    babi_val_txt = find_and_parse_story(babi_val_txt_in, period=True)
    babi_test_txt = find_and_parse_story(babi_test_txt_in, period=True)

    TEXT.build_vocab(babi_train_txt)
    opt.src_vocab_size = len(TEXT.vocab)
    opt.trg_vocab_size = len(TEXT.vocab)

    batch_size = 20
    eval_batch_size = 10

    size_tgt = 24  # 40000
    size_src = -1

    print('load train')
    babi_train_txt, babi_train_tgt, m_train = batchify_babi(
        babi_train_txt,
        batch_size,
        size_tgt=size_tgt,
        size_src=size_src,
        print_to_screen=False,
        separate_ques=True)

    print('load val')
    babi_val_txt, babi_val_tgt, m_val = batchify_babi(
        babi_val_txt,
        batch_size,
        size_tgt=size_tgt,
        size_src=size_src,
        separate_ques=True)

    print('load tst')
    babi_test_txt, babi_test_tgt, m_test = batchify_babi(
        babi_test_txt,
        batch_size,
        size_tgt=size_tgt,
        size_src=size_src,
        separate_ques=True)

    if False:
        if all((opt.train_path, opt.val_path) or int(opt.task) > 0):
            training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
        elif opt.data_pkl:
            training_data, validation_data = prepare_dataloaders(opt, device)
        else:
            raise

    training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)

    print(opt)

    transformer = Transformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0, opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt)


def prepare_dataloaders_from_bpe_files(opt, device):
    global babi_train_txt, babi_val_txt, babi_test_txt

    batch_size = opt.batch_size
    MIN_FREQ = 2
    MAX_LEN = 70
    if not opt.embs_share_weight:
        raise
    '''
    data = pickle.load(open(opt.data_pkl, 'rb'))
    MAX_LEN = data['settings'].max_len
    field = data['vocab']
    fields = (field, field)
    

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    train = TranslationDataset(
        fields=fields,
        path=opt.train_path, 
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)
    val = TranslationDataset(
        fields=fields,
        path=opt.val_path, 
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)
    '''


    opt.max_token_seq_len = MAX_LEN + 2
    opt.src_pad_idx = opt.trg_pad_idx =  TEXT.vocab.stoi[Constants.PAD_WORD]
    #opt.src_vocab_size = opt.trg_vocab_size = len(field.vocab)

    #train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    #val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    train_iterator = BucketIterator(
        babi_train_txt,
        batch_size=batch_size,
        device=device,
        sort_key=lambda x: torchtext.data.interleave_keys(len(x.src), len(x.trg)),
        train=True
    )

    val_iterator = BucketIterator(
        babi_val_txt,
        batch_size=batch_size,
        device=device,
        sort_key = lambda x: torchtext.data.interleave_keys(len(x.src), len(x.trg))
    )

    print(train_iterator, 'ti')

    return train_iterator, val_iterator


def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator


if __name__ == '__main__':
    main()
