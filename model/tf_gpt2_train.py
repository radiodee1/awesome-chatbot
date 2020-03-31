#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./train --dataset <file|directory|glob>


import argparse
import json
import os
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import time
import tqdm
from tensorflow.core.protobuf import rewriter_config_pb2
import random
import sys

sys.path.append('tf_gpt2/src/')

import model, sample, encoder
from encoder import Encoder

from load_dataset import load_dataset, Sampler
from accumulate import AccumulatingOptimizer
import memory_saving_gradients

from settings import hparams as hp

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = hp['save_dir'] + '/' + 'tf_gpt2_samples/'

model_name = hp['data_dir'] + '/' + 'tf_gpt2_data/'
checkpoint_dir = hp['save_dir'] + '/' + 'tf_gpt2_saved/'
CHECKPOINT_DIR = checkpoint_dir

HIDDEN_SIZE = 1024
GENERATE_SIZE = 10

parser = argparse.ArgumentParser(
    description='Fine-tune GPT-2 on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')

parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=0.000001, help='Learning rate for Adam')
parser.add_argument('--accumulate_gradients', metavar='N', type=int, default=1, help='Accumulate gradients across N minibatches.')
parser.add_argument('--memory_saving_gradients', default=False, action='store_true', help='Use gradient checkpointing to reduce vram usage.')
parser.add_argument('--only_train_transformer_layers', default=False, action='store_true', help='Restrict training to the transformer blocks.')

parser.add_argument('--train_special', action='store_true', help='test special training routine for babi stories')

parser.add_argument('--restore_from', type=str, default='latest', help='Either "latest", "fresh", or a path to a checkpoint file')
parser.add_argument('--run_name', type=str, default='run1', help='Run id. Name of subdirectory in checkpoint/ and samples/')
parser.add_argument('--sample_every', metavar='N', type=int, default=100, help='Generate samples every N steps')
parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=50, help='Sample this many tokens')
parser.add_argument('--sample_num', metavar='N', type=int, default=1, help='Generate this many samples')
parser.add_argument('--save_every', metavar='N', type=int, default=1000, help='Write a checkpoint every N steps')

parser.add_argument('--val_dataset', metavar='PATH', type=str, default=None, help='Dataset for validation loss, defaults to --dataset.')
parser.add_argument('--val_batch_size', metavar='SIZE', type=int, default=1, help='Batch size for validation.')
parser.add_argument('--val_batch_count', metavar='N', type=int, default=40, help='Number of batches for validation.')
parser.add_argument('--val_every', metavar='STEPS', type=int, default=0, help='Calculate validation loss every STEPS steps.')
parser.add_argument('--stop_after', metavar='STOP', type=int, default=100, help='Stop after training counter reaches STOP')

class SamplerVal(object):

    def __init__(self, chunks, encoder=None, char='\n', skip_delimeter=True):
        char = encoder.encode(char)
        tab = encoder.encode('\t')
        chunks = chunks[0]
        l = []
        self.chunks = []
        for i in chunks:
            #if i == tab[0]: i = chunks[0]
            if (i != char[0] and i != tab[0]) or not skip_delimeter:
                l.append(i)
            else:
                l.append(encoder.encode(' ')[0])
                #l.append(encoder.encode('.')[0])
            if i == char[0]:
                self.chunks.append(l)
                l = []
        self.total_size = len(self.chunks)

    def get(self, index):
        return self.chunks[index]

def name_parts(name):
    n = name.split('/')
    base = '.'.join(n[-1].split('.')[:-1])
    path = '/'.join(n[:-1]) + '/'
    # print(n,'n', path, 'p', base, 'base')
    from_name = base + '.from'
    to_name = base + '.to'
    ques_name = base + '.ques'
    return path + from_name, path + ques_name, path + to_name


def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

def get_encoder(model_name):
    with open(os.path.join(model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )


def main():
    args = parser.parse_args()
    try:
        logdir = os.path.join(CHECKPOINT_DIR, args.run_name)
        with open('logdir.txt', 'w') as z:
            z.write(logdir)
    except:
        pass
    enc = get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join(model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if args.sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    if args.model_name == '345M':
        args.memory_saving_gradients = True
        args.only_train_transformer_layers = True

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    with tf.Session(config=config) as sess:
        context = tf.placeholder(tf.int32, [args.batch_size, None])
        output = model.model(hparams=hparams, X=context)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))

        if args.val_every > 0:
            val_context = tf.placeholder(tf.int32, [args.val_batch_size, None])
            val_output = model.model(hparams=hparams, X=val_context)
            val_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=val_context[:, 1:], logits=val_output['logits'][:, :-1]))
            val_loss_summary = tf.summary.scalar('val_loss', val_loss)


        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=args.sample_length,
            context=context,
            batch_size=args.batch_size,
            temperature=1.0,
            top_k=40)

        all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        train_vars = [v for v in all_vars if '/h' in v.name] if args.only_train_transformer_layers else all_vars
        if args.accumulate_gradients > 1:
            if args.memory_saving_gradients:
                exit("Memory saving gradients are not implemented for gradient accumulation yet.")
            opt = AccumulatingOptimizer(
                opt=tf.train.AdamOptimizer(learning_rate=args.learning_rate),
                var_list=train_vars)
            opt_reset = opt.reset()
            opt_compute = opt.compute_gradients(loss)
            opt_apply = opt.apply_gradients()
            summary_loss = tf.summary.scalar('loss', opt_apply)
        else:
            opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            if args.memory_saving_gradients:
                opt_grads = memory_saving_gradients.gradients(loss, train_vars)
            else:
                opt_grads = tf.gradients(loss, train_vars)
            opt_grads = list(zip(opt_grads, train_vars))
            opt_apply = opt.apply_gradients(opt_grads)
            summary_loss = tf.summary.scalar('loss', loss)

        summary_log = tf.summary.FileWriter(
            os.path.join(CHECKPOINT_DIR, args.run_name))

        saver = tf.train.Saver(
            var_list=all_vars,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=2)
        sess.run(tf.global_variables_initializer())

        if args.restore_from == 'latest':
            ckpt = tf.train.latest_checkpoint(
                os.path.join(CHECKPOINT_DIR, args.run_name))
            if ckpt is None:
                # Get fresh GPT weights if new run.
                ckpt = tf.train.latest_checkpoint(
                    os.path.join(model_name))
        elif args.restore_from == 'fresh':
            ckpt = tf.train.latest_checkpoint(
                os.path.join(model_name))
        else:
            ckpt = tf.train.latest_checkpoint(args.restore_from)
        print('Loading checkpoint', ckpt)
        saver.restore(sess, ckpt)

        #print('Loading dataset...')
        #chunks = load_dataset(enc, args.dataset, args.combine)
        #data_sampler = Sampler(chunks)
        print('Loading train dataset...')
        from_name, ques_name, to_name = name_parts(args.dataset)

        trn_chunks_from = load_dataset(enc, from_name, args.combine) #if args.dataset else chunks
        #trn_chunks_ques = load_dataset(enc, ques_name, args.combine) if args.dataset else chunks
        trn_chunks_to = load_dataset(enc, to_name, args.combine) #if args.dataset else chunks

        skip_delimeter = True
        char = '\t'
        trn_data_sampler_from = SamplerVal(trn_chunks_from, enc, char=char, skip_delimeter=skip_delimeter)
        #trn_data_sampler_ques = SamplerVal(trn_chunks_ques, enc, char=char, skip_delimeter=skip_delimeter)
        trn_data_sampler_to = SamplerVal(trn_chunks_to, enc, char=char, skip_delimeter=skip_delimeter)

        len_v = 0
        data_sampler = []
        for i in range(trn_data_sampler_from.total_size):
            v = (
                    #enc.encode('\nQ: ') +
                    trn_data_sampler_from.get(i) +
                    #enc.encode('. \nA: ') +
                    trn_data_sampler_to.get(i) #  +
                    #enc.encode('. ')
            )

            v = v[: HIDDEN_SIZE - 1]
            len_v += len(v)
            #data_sampler.extend(v) ##
            data_sampler.append(v)
            pass

        if len_v < HIDDEN_SIZE:
            mult = HIDDEN_SIZE // len_v + 1
            for i in range(mult):
                x = data_sampler[:]
                data_sampler.extend(x)
            data_sampler = Sampler([np.array(data_sampler)] )

        #if not args.train_special and len_v >= HIDDEN_SIZE:
        #    data_sampler = Sampler([np.array(data_sampler)])

        if args.val_every > 0 and False:
            val_chunks = load_dataset(enc, args.val_dataset, args.combine) if args.val_dataset else chunks
        if not isinstance(data_sampler, list):
            print('dataset has', data_sampler.total_size, 'tokens')
        print('Training...')

        if args.val_every > 0:
            # Sample from validation set once with fixed seed to make
            # it deterministic during training as well as across runs.
            val_data_sampler = Sampler(val_chunks, seed=1)
            val_batches = [[val_data_sampler.sample(1024) for _ in range(args.val_batch_size)]
                           for _ in range(args.val_batch_count)]

        counter = 1
        counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')
        if os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, 'r') as fp:
                counter = int(fp.read()) + 1

        def save():
            maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
            print(
                'Saving',
                os.path.join(CHECKPOINT_DIR, args.run_name,
                             'model-{}').format(counter))
            saver.save(
                sess,
                os.path.join(CHECKPOINT_DIR, args.run_name, 'model'),
                global_step=counter)
            with open(counter_path, 'w') as fp:
                fp.write(str(counter) + '\n')

            #print(model_name, 'mn')
            GPT2_DIR_X = model_name
            cd = CHECKPOINT_DIR + "/" + args.run_name
            if not os.path.isfile(cd + '/' + 'encoder.json'):
                os.system("cp " + GPT2_DIR_X + '/' + 'encoder.json ' + cd + '/.')
                os.system('cp ' + GPT2_DIR_X + "/" + 'vocab.bpe ' + cd + '/.')

        def generate_samples():
            print('Generating samples...')
            #context_tokens = data_sampler.sample(1)
            #context_tokens = data_sampler[0]
            context_tokens = trn_data_sampler_from.get(random.randint(0,trn_data_sampler_from.total_size))
            #print(enc.decode(context_tokens), len(context_tokens))
            #print(args.batch_size * [context_tokens])

            all_text = []
            index = 0
            while index < args.sample_num:
                out = sess.run(
                    tf_sample,
                    feed_dict={context: args.batch_size * [context_tokens]})
                for i in range(min(args.sample_num - index, args.batch_size)):
                    text = enc.decode(out[i])
                    text = '======== SAMPLE {} ========\n{}\n'.format(
                        index + 1, text)
                    all_text.append(text)
                    index += 1
            print(text)
            maketree(os.path.join(SAMPLE_DIR, args.run_name))
            with open(
                    os.path.join(SAMPLE_DIR, args.run_name,
                                 'samples-{}').format(counter), 'w') as fp:
                fp.write('\n'.join(all_text))

        def validation():
            print('Calculating validation loss...')
            losses = []
            for batch in tqdm.tqdm(val_batches):
                losses.append(sess.run(val_loss, feed_dict={val_context: batch}))
            v_val_loss = np.mean(losses)
            v_summary = sess.run(val_loss_summary, feed_dict={val_loss: v_val_loss})
            summary_log.add_summary(v_summary, counter)
            summary_log.flush()
            print(
                '[{counter} | {time:2.2f}] validation loss = {loss:2.2f}'
                .format(
                    counter=counter,
                    time=time.time() - start_time,
                    loss=v_val_loss))

        def sample_batch():
            #z = [data_sampler.sample(1024) for _ in range(args.batch_size)]
            #print(len(data_sampler))
            #print(len(data_sampler[0]))
            z = [data_sampler[random.randint(0, args.batch_size)]]
            #print(enc.decode(z[0]))
            #print(z[1],'\n1' ,z[2],'\n2' ,z[3] ,len(data_sampler[0]))
            #exit()
            return z


        avg_loss = (0.0, 0.0)
        start_time = time.time()

        try:
            while counter != args.stop_after:
                if counter % args.save_every == 0:
                    save()
                if counter % args.sample_every == 0:
                    generate_samples()
                    pass
                if args.val_every > 0 and (counter % args.val_every == 0 or counter == 1):
                    validation()

                if args.accumulate_gradients > 1:
                    sess.run(opt_reset)
                    for _ in range(args.accumulate_gradients):
                        sess.run(
                            opt_compute, feed_dict={context: sample_batch()})
                    (v_loss, v_summary) = sess.run((opt_apply, summary_loss))
                else:
                    (_, v_loss, v_summary) = sess.run(
                        (opt_apply, loss, summary_loss),
                        feed_dict={context: sample_batch()})

                summary_log.add_summary(v_summary, counter)

                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))

                counter += 1
        except KeyboardInterrupt:
            print('\ninterrupted')

        finally:
            save()


if __name__ == '__main__':
    main()
