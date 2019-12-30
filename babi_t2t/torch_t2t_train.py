#!/usr/bin/python3


"""
Sequence-to-Sequence Modeling with nn.Transformer and TorchText
===============================================================

This is a tutorial on how to train a sequence-to-sequence model
that uses the
`nn.Transformer <https://pytorch.org/docs/master/nn.html?highlight=nn%20transformer#torch.nn.Transformer>`__ module.

PyTorch 1.2 release includes a standard transformer module based on the
paper `Attention is All You
Need <https://arxiv.org/pdf/1706.03762.pdf>`__. The transformer model
has been proved to be superior in quality for many sequence-to-sequence
problems while being more parallelizable. The ``nn.Transformer`` module
relies entirely on an attention mechanism (another module recently
implemented as `nn.MultiheadAttention <https://pytorch.org/docs/master/nn.html?highlight=multiheadattention#torch.nn.MultiheadAttention>`__) to draw global dependencies
between input and output. The ``nn.Transformer`` module is now highly
modularized such that a single component (like `nn.TransformerEncoder <https://pytorch.org/docs/master/nn.html?highlight=nn%20transformerencoder#torch.nn.TransformerEncoder>`__
in this tutorial) can be easily adapted/composed.

.. image:: ../_static/img/transformer_architecture.jpg

"""

######################################################################
# Define the model
# ----------------
#


######################################################################
# In this tutorial, we train ``nn.TransformerEncoder`` model on a
# language modeling task. The language modeling task is to assign a
# probability for the likelihood of a given word (or a sequence of words)
# to follow a sequence of words. A sequence of tokens are passed to the embedding
# layer first, followed by a positional encoding layer to account for the order
# of the word (see the next paragraph for more details). The
# ``nn.TransformerEncoder`` consists of multiple layers of
# `nn.TransformerEncoderLayer <https://pytorch.org/docs/master/nn.html?highlight=transformerencoderlayer#torch.nn.TransformerEncoderLayer>`__. Along with the input sequence, a square
# attention mask is required because the self-attention layers in
# ``nn.TransformerEncoder`` are only allowed to attend the earlier positions in
# the sequence. For the language modeling task, any tokens on the future
# positions should be masked. To have the actual words, the output
# of ``nn.TransformerEncoder`` model is sent to the final Linear
# layer, which is followed by a log-Softmax function.
#


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
#import pprint
#pp = pprint.PrettyPrinter(indent=4)

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        #self.softmax = nn.Softmax(dim=2)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        #output = self.softmax(output)
        #output = torch.argmax(output, dim=-1) ## -3
        return output


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


######################################################################
# Load and batch data
# -------------------
#


######################################################################
# The training process uses Wikitext-2 dataset from ``torchtext``. The
# vocab object is built based on the train dataset and is used to numericalize
# tokens into tensors. Starting from sequential data, the ``batchify()``
# function arranges the dataset into columns, trimming off any tokens remaining
# after the data has been divided into batches of size ``batch_size``.
# For instance, with the alphabet as the sequence (total length of 26)
# and a batch size of 4, we would divide the alphabet into 4 sequences of
# length 6:
#
# .. math::
#   \begin{bmatrix}
#   \text{A} & \text{B} & \text{C} & \ldots & \text{X} & \text{Y} & \text{Z}
#   \end{bmatrix}
#   \Rightarrow
#   \begin{bmatrix}
#   \begin{bmatrix}\text{A} \\ \text{B} \\ \text{C} \\ \text{D} \\ \text{E} \\ \text{F}\end{bmatrix} &
#   \begin{bmatrix}\text{G} \\ \text{H} \\ \text{I} \\ \text{J} \\ \text{K} \\ \text{L}\end{bmatrix} &
#   \begin{bmatrix}\text{M} \\ \text{N} \\ \text{O} \\ \text{P} \\ \text{Q} \\ \text{R}\end{bmatrix} &
#   \begin{bmatrix}\text{S} \\ \text{T} \\ \text{U} \\ \text{V} \\ \text{W} \\ \text{X}\end{bmatrix}
#   \end{bmatrix}
#
# These columns are treated as independent by the model, which means that
# the dependence of ``G`` and ``F`` can not be learned, but allows more
# efficient batch processing.
#

parser = argparse.ArgumentParser(
    description='Fine-tune a Transformer.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--tenk', action='store_true', help='use ten-k dataset')
parser.add_argument('--task', default=1, help='use specific question-set/task')
parser.add_argument('--lr', default=0.1, help='learning rate', type=float)
parser.add_argument('--epochs', default=30, help='number of epochs', type=int)
parser.add_argument('--no_scheduler', action='store_false',help='cancel learning rate decay')
args = parser.parse_args()

import torchtext

from torchtext.data.utils import get_tokenizer
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
#train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
#TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ten_k = args.tenk
task = args.task #1 #9
babi_train_txt, babi_val_txt, babi_test_txt = torchtext.datasets.BABI20.splits(TEXT,  root='../raw/', tenK=ten_k, task=task)

def find_and_parse_story(data, period=False):
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


babi_train_txt = find_and_parse_story(babi_train_txt, period=True)
babi_val_txt = find_and_parse_story(babi_val_txt, period=True)
babi_test_txt = find_and_parse_story(babi_test_txt, period=True)

TEXT.build_vocab(babi_train_txt)

def batchify_babi(data, bsz, separate_ques=True, size_src=200, size_tgt=200, print_to_screen=False):
    new_data = []
    target_data = []
    for ii in range(len(data.examples)):
        z = data.examples[ii]
        target_data_tmp = []
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
            z.story.extend(z.query)
            z.story.extend([ '<eos>'])
            #z.story.extend('.')
            new_data.append(z.story)
            target_data.append([z.answer])
        pass
    if print_to_screen: print(new_data,'nd')
    m = max(len(x) for x in new_data)
    n = max(len(x[0]) for x in target_data)
    m = max(m, size_src)
    #n = max(n, size_tgt)
    n = m
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

        padded_data = torch.zeros(1, m, dtype=torch.long)
        padded_target = torch.zeros(1, n, dtype=torch.long)
        new_n_data = torch.zeros( len(new_data), m, dtype=torch.long)
        target_n_data = torch.zeros( len(target_data), n, dtype=torch.long)

        for jj in range(len(new_data)):
            ## do source ##
            z = TEXT.numericalize([new_data[jj]])
            if z.size(0) > 1:
                z = z.t()
            p = padded_data[:]
            p[0, :z.size(1)] = z
            new_n_data[jj, :] = p
            ## do target ##
            y = TEXT.numericalize(target_data[jj])
            if y.size(0) > 1:
                y = y.t()
            q = padded_target[:]
            q[0,:len(y[0])] = y
            target_n_data[jj, :] = q

        new_n_data = new_n_data.t().contiguous()
        target_n_data = target_n_data.t().contiguous()

    return new_n_data.to(device), target_n_data.to(device)


def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10

size_tgt = 24 #40000
size_src = -1
babi_train_txt, babi_train_tgt = batchify_babi(babi_train_txt, batch_size,size_tgt=size_tgt, size_src=size_src, separate_ques=True)
babi_val_txt, babi_val_tgt = batchify_babi(babi_val_txt, batch_size, size_tgt=size_tgt, size_src=size_src, separate_ques=True)
babi_test_txt, babi_test_tgt = batchify_babi(babi_test_txt, batch_size, size_tgt=size_tgt, size_src=size_src, separate_ques=True)

######################################################################
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# ``get_batch()`` function generates the input and target sequence for
# the transformer model. It subdivides the source data into chunks of
# length ``bptt``. For the language modeling task, the model needs the
# following words as ``Target``. For example, with a ``bptt`` value of 2,
# we’d get the following two Variables for ``i`` = 0:
#
# .. image:: ../_static/img/transformer_input_target.png
#
# It should be noted that the chunks are along dimension 0, consistent
# with the ``S`` dimension in the Transformer model. The batch dimension
# ``N`` is along dimension 1.
#

bptt = 35
def get_batch_babi(source, target, i, print_to_screen=False, bptt=35, flatten_target=True):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = target[i:i + seq_len]
    if flatten_target:
        target = target.view(-1)
    if print_to_screen: print(data, target, i, 'dti')
    return data, target

#bptt = 35
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def show_strings(source):
    if len(source.size()) > 1:
        source = source.squeeze(0)
    for i in source:
        if i != 0:
            print(TEXT.vocab.itos[i], end=' | ')
    print()

if False:
    tt1, tt2 = get_batch_babi(babi_val_txt, babi_val_tgt, 0, flatten_target=False)

    print(tt1,'\n',tt2)
    print(tt1.size(), tt2.size(),'t,t')
    #show_strings(babi_train_txt[0])
    #show_strings(babi_train_tgt[0])

    show_strings(tt1.t()[0])
    print()
    show_strings(tt2.t()[0])
    exit()


def show_tensor_vals(source):
    zero = 0
    for i in range(source.size(0)):
        for ii in range(len(source[i])):
            z = source[i][ii]
            if not z is 0:
                print(z, end='|')
                print(TEXT.vocab.itos[z], end='|')
            else:
                z += 1
        pass
    print('\n',zero, 'zeros')


######################################################################
# Initiate an instance
# --------------------
#


######################################################################
# The model is set up with the hyperparameter below. The vocab size is
# equal to the length of the vocab object.
#

ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)


######################################################################
# Run the model
# -------------
#


######################################################################
# `CrossEntropyLoss <https://pytorch.org/docs/master/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss>`__
# is applied to track the loss and
# `SGD <https://pytorch.org/docs/master/optim.html?highlight=sgd#torch.optim.SGD>`__
# implements stochastic gradient descent method as the optimizer. The initial
# learning rate is set to 5.0. `StepLR <https://pytorch.org/docs/master/optim.html?highlight=steplr#torch.optim.lr_scheduler.StepLR>`__ is
# applied to adjust the learn rate through epochs. During the
# training, we use
# `nn.utils.clip_grad_norm\_ <https://pytorch.org/docs/master/nn.html?highlight=nn%20utils%20clip_grad_norm#torch.nn.utils.clip_grad_norm_>`__
# function to scale all the gradient together to prevent exploding.
#

criterion = nn.CrossEntropyLoss()
lr = args.lr # 1.0 #5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

label = 'val'

import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, babi_train_txt.size(0) - 1, bptt)):
        data, targets = get_batch_babi(babi_train_txt, babi_train_tgt, i)
        bsz = data.size(0)
        optimizer.zero_grad()
        output = model(data)

        #predictions = output
        prediction_text = torch.argmax(output.view(-1,ntokens), dim=1)

        if (not ten_k or i % 100 == 0) and True:
            print(
                TEXT.vocab.itos[prediction_text[-3].item()],
                TEXT.vocab.itos[prediction_text[-2].item()],
                TEXT.vocab.itos[prediction_text[-1].item()],
                '['+TEXT.vocab.itos[targets[-1].item()]+']')
            print( output.size(), targets.size(), targets[0].item(),prediction_text[-1].item(), 'p,tt')

        loss = criterion(output.view( -1, ntokens), targets) ### <---
        #loss = criterion(output.view(-1, ntokens), targets) ### <---
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(babi_train_txt) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source, data_tgt, show_accuracy=False):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    acc = 0
    saved_dim = -1
    out_dim = -1
    bptt = 1
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch_babi(data_source, data_tgt, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            output_flat_t = output.transpose(1,0).contiguous().view(-1, ntokens)
            output_argmax = torch.argmax(output_flat_t, dim=-1)

            total_loss += len(data) * criterion(output_flat, targets).item()

            targets_text = targets[-1].item()
            if False:
                print(targets.size(), torch.argmax(output_flat_t, dim=-1)[:20], 'tt,out', label)
                print(TEXT.vocab.itos[output_argmax[0].item()], 'itos', output_argmax[0].item(),'sd', saved_dim,i)

            out_dim = output.size(0)

            if saved_dim == -1 or saved_dim == out_dim:
                saved_dim = out_dim
                if i == 0: print(out_dim, 'dim ', end='|')
                for ii in range(0, 10): #output_flat.size(0)):
                    text = torch.argmax(output_flat_t, dim=-1)[ii].item()
                    if text != 0:
                        print(TEXT.vocab.itos[text], end='|')
                    if text == targets_text and text != 0:
                        acc += 1
                        print(TEXT.vocab.itos[text],'score acc')
                        break
                if i == 0: print()
    if show_accuracy:
        print('acc:', acc / len(data_source) * 100.00)
    return total_loss / (len(data_source) - 1)

######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.

best_val_loss = float("inf")
epochs = args.epochs # 30 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    label = 'val'
    val_loss = evaluate(model, babi_val_txt, babi_val_tgt)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    label = 'tst'
    evaluate(model, babi_test_txt, babi_test_tgt, show_accuracy=True)

    if not args.no_scheduler: scheduler.step()


######################################################################
# Evaluate the model with the test dataset
# -------------------------------------
#
# Apply the best model to check the result with the test dataset.

test_loss = evaluate(best_model, babi_test_txt, babi_test_tgt)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
