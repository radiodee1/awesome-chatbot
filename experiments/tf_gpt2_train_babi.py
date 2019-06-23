#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./train --dataset <file|directory|glob>


import argparse
import json
import os
import numpy as np
import tensorflow as tf
import time
import tqdm
from tensorflow.core.protobuf import rewriter_config_pb2
import sys

sys.path.append('../model/tf_gpt2/src/')
sys.path.append('../model/')

import model, sample, encoder
from encoder import Encoder

from load_dataset import load_dataset, Sampler
from accumulate import AccumulatingOptimizer
import memory_saving_gradients

from settings import hparams as hp
import tensorflow.contrib.slim as slim
import datetime
import json

HIDDEN_SIZE = 1024 -1

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = hp['save_dir'] + '/' + 'tf_gpt2_samples/'

model_name =  hp['data_dir'] + '/' + 'tf_gpt2_data/'
checkpoint_dir = hp['save_dir'] + '/' + 'tf_gpt2_saved/'
CHECKPOINT_DIR = checkpoint_dir

parser = argparse.ArgumentParser(
    description='Fine-tune GPT-2 on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, default='../data/train.from', help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')

parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=0.0001, help='Learning rate for Adam')
parser.add_argument('--accumulate_gradients', metavar='N', type=int, default=1, help='Accumulate gradients across N minibatches.')
parser.add_argument('--memory_saving_gradients', default=False, action='store_true', help='Use gradient checkpointing to reduce vram usage.')
parser.add_argument('--only_train_transformer_layers', default=False, action='store_true', help='Restrict training to the transformer blocks.')

parser.add_argument('--restore_from', type=str, default='latest', help='Either "latest", "fresh", or a path to a checkpoint file')
parser.add_argument('--run_name', type=str, default='run1', help='Run id. Name of subdirectory in checkpoint/ and samples/')
parser.add_argument('--sample_every', metavar='N', type=int, default=100, help='Generate samples every N steps')
parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=25, help='Sample this many tokens')
parser.add_argument('--sample_num', metavar='N', type=int, default=1, help='Generate this many samples')
parser.add_argument('--save_every', metavar='N', type=int, default=1000, help='Write a checkpoint every N steps')

parser.add_argument('--val_dataset', metavar='PATH', type=str, default='../data/valid.from', help='Dataset for validation loss, defaults to --dataset.')
parser.add_argument('--val_batch_size', metavar='SIZE', type=int, default=40, help='Batch size for validation.')
parser.add_argument('--val_batch_count', metavar='N', type=int, default=-1, help='Number of batches for validation.')
parser.add_argument('--val_every', metavar='STEPS', type=int, default=500, help='Calculate validation loss every STEPS steps.')
parser.add_argument('--stop_after', metavar='STOP', type=int, default=None, help='Stop after training counter reaches STOP')


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

def name_parts(name):
    n = name.split('/')
    base = '.'.join(n[-1].split('.')[:-1])
    path = '/'.join(n[:-1]) + '/'
    # print(n,'n', path, 'p', base, 'base')
    from_name = base + '.from'
    to_name = base + '.to'
    ques_name = base + '.ques'
    return path + from_name, path + ques_name, path + to_name

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

class SamplerVal(object):

    def __init__(self, chunks, encoder=None, char='\n', skip_delimeter=True):
        char = encoder.encode(char)
        chunks = chunks[0]
        l = []
        self.chunks = []
        for i in chunks:
            if i != char[0] or not skip_delimeter:
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

def main():
    args = parser.parse_args()
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

    acc_total = 0
    acc_over_time = []
    loss_avg_over_time = []

    if args.val_every > 0:

        # val_context = tf.placeholder(tf.int32, [args.val_batch_size, None])
        val_context = tf.placeholder(np.int32, [ 1, None])

        val_output = model.model(hparams=hparams, X=val_context)
        val_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=val_context[:, 1:], logits=val_output['logits'][:, :-1]))
        val_loss_summary = tf.summary.scalar('val_loss', val_loss)


        tf_sample_val = sample.sample_sequence(
            hparams=hparams,
            length=1, #args.sample_length,
            context=val_context,
            batch_size=1, #args.batch_size,
            temperature=10.001,
            top_k=1)

    with tf.Session(config=config) as sess:
        context = tf.placeholder(tf.int32, [args.batch_size, None])
        output = model.model(hparams=hparams, X=context)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))

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

        print('Loading train dataset...')
        from_name, ques_name, to_name = name_parts( args.dataset ) #'../data/train.from')

        trn_chunks_from = load_dataset(enc, from_name, args.combine) if args.val_dataset else chunks
        trn_chunks_ques = load_dataset(enc, ques_name, args.combine) if args.val_dataset else chunks
        trn_chunks_to = load_dataset(enc, to_name, args.combine) if args.val_dataset else chunks


        skip_delimeter = True
        trn_data_sampler_from = SamplerVal(trn_chunks_from, enc, skip_delimeter=skip_delimeter)
        trn_data_sampler_ques = SamplerVal(trn_chunks_ques, enc, skip_delimeter=skip_delimeter)
        trn_data_sampler_to = SamplerVal(trn_chunks_to, enc, skip_delimeter=skip_delimeter)

        data_sampler = []
        for i in range(trn_data_sampler_from.total_size):
            v = (
                    trn_data_sampler_from.get(i) +
                    trn_data_sampler_ques.get(i) +
                    enc.encode('. ')  +
                    trn_data_sampler_to.get(i)  +
                    enc.encode('<|endoftext|>')
            )
            # v += [enc.encode(' ')[0] for _ in range(HIDDEN_SIZE - len(v) )]
            data_sampler.append(v)
            pass

        #chunks = load_dataset(enc, args.dataset, args.combine)
        data_sampler = Sampler([np.array(data_sampler)])

        if args.val_every > 0:
            print('Loading validation dataset...')
            #val_chunks = load_dataset(enc, args.val_dataset, args.combine) if args.val_dataset else chunks

            from_name, ques_name, to_name = name_parts(args.val_dataset)

            val_chunks_from = load_dataset(enc, from_name, args.combine) if args.val_dataset else chunks
            val_chunks_ques = load_dataset(enc, ques_name, args.combine) if args.val_dataset else chunks
            val_chunks_to =   load_dataset(enc, to_name,   args.combine) if args.val_dataset else chunks

        print('dataset has', data_sampler.total_size, 'tokens')
        print('Training...')

        if args.val_every > 0:

            val_data_sampler_from = SamplerVal(val_chunks_from, enc)
            val_data_sampler_ques = SamplerVal(val_chunks_ques, enc)
            val_data_sampler_to = SamplerVal(val_chunks_to, enc)

            if args.val_batch_count == -1:
                args.val_batch_count = val_data_sampler_from.total_size

            val_batches = []
            for i in range(args.val_batch_count):
                v = (
                        val_data_sampler_from.get(i) +
                        val_data_sampler_ques.get(i) +
                        enc.encode('. ')
                ) #+ val_data_sampler_to.get(i)

                #v += [enc.encode(' ')[0] for _ in range(HIDDEN_SIZE - len(v) )]
                val_batches.append(v)
                pass

        #exit()
        counter = 1
        counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')
        if os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, 'r') as fp:
                counter = int(fp.read()) + 1

        txt_file_path = os.path.join(CHECKPOINT_DIR, args.run_name, args.run_name + '.summary.txt')

        def save_summary():
            txt = ''

            if not os.path.exists(txt_file_path):
                a = vars(args)
                txt += 'Summary for ' + args.run_name + '\n'
                txt += str(datetime.datetime.now()) + '\n\n'
                txt += json.dumps(a) + '\n'
                txt += '-----\n'
                pass
            txt += str(datetime.datetime.now()) + '\n'
            txt += 'acc: ' + ', '.join([str(i) for i in acc_over_time]) + '\n'
            txt += 'loss: ' + ', '.join([str(i) for i in loss_avg_over_time]) + '\n'
            txt += 'counter: ' + str(counter) + '\n'
            txt += '-----\n'
            print(txt)
            with open(txt_file_path, 'a') as f:
                f.write(txt + '\n')

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
            save_summary()

        def generate_samples():
            print('Generating samples...')
            context_tokens = data_sampler.sample(1)
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

        def sample_batch():
            #print(enc.encode('<|endoftext|>'), 'eot')
            #print(data_sampler.sample(1024))
            return [data_sampler.sample(1024)[0] for _ in range(args.batch_size)]

        def validation_by_sample():
            print('Generating validation...')
            global acc_total
            if False:
                losses = []
                for batch in tqdm.tqdm(val_batches):
                    batch = np.reshape(batch, [1,-1])
                    v = sess.run(val_loss, feed_dict={val_context: batch})
                    #print(v, 'v')
                    losses.append(v)
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
            acc_total = 0
            generated = 0
            for _ in range(len(val_batches)):

                val_batches_in = val_batches[generated]
                context_tokens = np.reshape(val_batches_in, [ 1, -1])

                #print(context_tokens, 'ct1')
                for x in range(10):

                    out = sess.run(tf_sample_val, feed_dict={val_context: context_tokens})
                    #print(out[0][-1])
                    context_tokens = out

                compare = enc.decode(val_data_sampler_to.get(generated)) # + ' <|endoftext|>'
                compare = ' '.join(compare.split(' '))

                generated += 1

                text = enc.decode(out[0])

                if text.strip().endswith('.'): ## remove trailing period
                    text = text.strip()[:-1]

                if text.strip().endswith('<|endoftext|>'):
                    text = text.strip()[: - len('<|endoftext|>')]

                t_vals = text.split(' ')
                if '<' in t_vals[-1] or '>' in t_vals[-1]:
                    t_vals = t_vals[:-1]

                text = ' '.join(t_vals)

                if compare.strip().endswith('.'):
                    compare = compare.strip()[:-1]

                if compare.strip().endswith('<|endoftext|>'):
                    compare = compare.strip()[: - len('<|endoftext|>')]

                notification = ''
                len_bar = 40
                if text.strip().lower().endswith(compare.strip().lower()):
                    acc_total += 1
                    notification = 'SCORE!! '
                    len_bar = 40 - len(notification)

                print(notification + "=" * len_bar + " SAMPLE " + str(generated) + " " + "=" * len_bar + notification)
                print(text)
            print("=" * 80)
            return acc_total
            pass

        avg_loss = (0.0, 0.0)
        start_time = time.time()
        count_success = 0
        count_success_with_skips = 0
        acc = 0.0

        try:
            while counter != args.stop_after:
                #model_summary()

                if counter % args.save_every == 0:
                    save()
                if counter % args.sample_every == 0:
                    #generate_samples()
                    pass
                if args.val_every > 0 and (counter % args.val_every == 0 or counter == 1):
                    acc_total = validation_by_sample()
                    acc = acc_total / len(val_batches) * 100

                    acc_over_time.append(acc)
                    if avg_loss[1] > 0.0:
                        loss_avg_over_time.append(avg_loss[0] / avg_loss[1])
                    else:
                        loss_avg_over_time.append(0)

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

                if float(acc) is 100.0:
                    save()
                    print('validation accuracy 100', time.time() - start_time)
                    count_success += 1
                    count_success_with_skips += 1
                    if count_success >= 2 or count_success_with_skips >= 4:
                        exit()
                else:
                    count_success = 0

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]), 'acc='+str(acc), end=' ')
                print('total=' + str(acc_total))

                counter += 1
        except KeyboardInterrupt:
            print('interrupted')
        finally:
            save()


if __name__ == '__main__':
    main()
