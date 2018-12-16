#!/usr/bin/python3.6


import os as os
import sys
import argparse
from settings import hparams
import tokenize_weak

def save_to_file(pos_context, pos_ques, pos_ans, context_filename, ques_filename, ans_filename, lowercase=False):
    with open(context_filename, 'w') as z:
        for line in pos_context:
            #line = tokenize_weak.format(line)
            if lowercase: line = line.lower()
            z.write(line + '\n')

    with open(ques_filename,'w') as z:
        for line in pos_ques:
            z.write(line + '\n')

    with open(ans_filename, 'w') as z:
        for line in pos_ans:
            if lowercase: line = line.lower()
            z.write(line + '\n')



if __name__ == '__main__':
    print('do make train and test')

    parser = argparse.ArgumentParser(description='Make text files.')

    parser.add_argument('--file', help='file name with path for POS input.')
    parser.add_argument('--lowercase', help='record all input data in lowercase.', action='store_true')
    parser.add_argument('--split', help='float for split of valid/test files')
    parser.add_argument('--start', help='starting position')
    parser.add_argument('--length', help='length of corpus segment, (10000 default)')
    args = parser.parse_args()
    args = vars(args)
    print(args)

    args_start = 0
    args_length = 10000

    if args['start'] is not None:
        args_start = int(args['start'])

    if args['length'] is not None:
        args_length = int(args['length'])

    args_split_train = 0.9
    args_split_valid = 0.05
    args_split_test = 0.05

    if args['split'] is not None:
        args_split_test = float(args['split'])
        args_split_valid = float(args['split'])
        args_split_train = 1.0 - (args_split_valid + args_split_test)

    args_input_path = '../raw/' + 'ner_dataset.csv'

    args_train_question = str(
        hparams['data_dir'] +
        hparams['train_name'] + '.' +
        hparams['pos_name'] + '.' +
        hparams['question_ending']
    )

    args_train_answer = str(
        hparams['data_dir'] +
        hparams['train_name'] + '.' +
        hparams['pos_name'] + '.' +
        hparams['tgt_ending']
    )

    args_train_context = str(
        hparams['data_dir'] +
        hparams['train_name'] + '.' +
        hparams['pos_name'] + '.' +
        hparams['src_ending']
    )

    args_test_question = str(
        hparams['data_dir'] +
        hparams['test_name'] + '.' +
        hparams['pos_name'] + '.' +
        hparams['question_ending']
    )

    args_test_answer = str(
        hparams['data_dir'] +
        hparams['test_name'] + '.' +
        hparams['pos_name'] + '.' +
        hparams['tgt_ending']
    )

    args_test_context = str(
        hparams['data_dir'] +
        hparams['test_name'] + '.' +
        hparams['pos_name'] + '.' +
        hparams['src_ending']
    )

    args_valid_question = str(
        hparams['data_dir'] +
        hparams['valid_name'] + '.' +
        hparams['pos_name'] + '.' +
        hparams['question_ending']
    )

    args_valid_answer = str(
        hparams['data_dir'] +
        hparams['valid_name'] + '.' +
        hparams['pos_name'] + '.' +
        hparams['tgt_ending']
    )

    args_valid_context = str(
        hparams['data_dir'] +
        hparams['valid_name'] + '.' +
        hparams['pos_name'] + '.' +
        hparams['src_ending']
    )

    if args['file'] is not None:
        args_input_path = str(args['file'])

    args_lowercase = False
    if args['lowercase'] is True:
        args_lowercase = True

    pos_input = []
    pos_context = []
    pos_question = []
    pos_answer = []

    args_end_string = hparams['eol'] + ' ' + hparams['eol']

    if os.path.isfile(args_input_path):
        ''' read input '''
        with open(args_input_path, 'rb') as z:
            text = z.readlines()
            line = ''
            for xx in text[args_start: args_length]:
                if xx.endswith((b'\n', b'\r')):
                    line += xx.strip().decode('utf-8', errors='ignore')
                    pos_input.append(line.split(',')[:-1])
                    line = ''
                    #pos_input.append(xx)
                else:
                    line += xx.strip().decode('utf-8', errors='ignore')
        ''' parse input '''
        context_string = ''
        for line in pos_input[1:]:

            if line[0] != '':
                if len(context_string) > 1:
                    pos_context.append(args_end_string)
                    pos_question.append(args_end_string)
                    pos_answer.append(args_end_string)

                context_string = '' # line[1]

            if True:
                if len(context_string) > 0:
                    context_string += ' '

                context_string += line[1]

                context_string = context_string.replace('"', '')
                context_string = context_string.replace("'", '')

                pos_context.append(context_string)
                pos_question.append(hparams['unk'])
                pos_answer.append(line[2])

                #print(line[2],'line 2')
            pass
        ''' save files '''

        idx_start_0_train = int(len(pos_answer) * args_split_train)
        idx_start_1_test = int(len(pos_answer) * (args_split_test + args_split_train))
        idx_start_2_valid = int(len(pos_answer) * (args_split_test + args_split_train))

        print(idx_start_0_train)
        print(idx_start_1_test)
        #print(idx_start_2_valid)
        print(len(pos_answer) - idx_start_2_valid)

        save_to_file(
            pos_context[:idx_start_0_train],
            pos_question[:idx_start_0_train],
            pos_answer[:idx_start_0_train],
            args_train_context,
            args_train_question,
            args_train_answer,
            lowercase=args_lowercase
        )
        save_to_file(
            pos_context[idx_start_0_train:idx_start_1_test],
            pos_question[idx_start_0_train:idx_start_1_test],
            pos_answer[idx_start_0_train:idx_start_1_test],
            args_test_context,
            args_test_question,
            args_test_answer,
            lowercase=args_lowercase
        )
        save_to_file(
            pos_context[idx_start_2_valid:],
            pos_question[idx_start_2_valid:],
            pos_answer[idx_start_2_valid:],
            args_valid_context,
            args_valid_question,
            args_valid_answer,
            lowercase=args_lowercase
        )
    else:
        print('bad file or directory.')

