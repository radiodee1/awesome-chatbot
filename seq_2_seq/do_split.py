#!/usr/bin/env python3

import argparse
import os
import sys
import random
import xml.etree.ElementTree as ET
#from tokenize_weak import format

def format(input):
    return input.lower()

hparams = {

    'save_dir': "./data",
    'data_dir': "./data",
    'test_name': "test",
    'train_name': "train",
    'valid_name':'valid',
    'src_ending': "from",
    'tgt_ending': "to",
    'question_ending':'ques',
    'hist_ending': 'hist',
    'babi_name':'babi',
    'eol': 'eol',
    'unk': 'unk',
    'sol': 'sol'
}

xml_list = []
xml_freq = []
root = None

pronouns = [
    'name',
    'names',
    #'my',
    #'your',
    #'his',
    #'her',
    #'their',

]

def add_to_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()
    x_list = []
    x_freq = []
    for x in root:
        sub_dict = {}

        for e in x:
            sub_dict[e.tag] = e.text.strip()
            if e.tag == 'use_every':
                z = float(e.text.strip())
                if z < 1:
                    x_freq.append(float(e.text.strip()))
                else:
                    x_freq.append(int(e.text.strip()))
        x_list.append(sub_dict)
    print(x_list, x_freq)
    return x_list, x_freq
    pass

def insert_xml(iter, from_handle, to_handle, question_handle):
    for i in xml_list:
        if iter % int(i['use_every']) == 0:
            from_handle.write(i['from'] + '\n')
            to_handle.write(i['to'] + '\n')
            question_handle.write(i['question'] + '\n')
    return

def stop_repeats(lst):
    j = []
    k = ''
    for i in lst.split():
        if k == i:
            continue
        j.append(i)
        k = i

    return ' '.join(j)

def move_order(first, second):
    mid = ''
    first = first.strip()
    second = second.strip()
    if second.endswith('?'):
        mid = second[:]
        second = first[:]
        first = mid[:]
    return first, second

def sentence_contains(sentence, words):
    test = False
    list_in = sentence.split()
    for i in list_in:
        if i.lower() in words: test = True
    return test

if __name__ == '__main__':
    tokenizer = None

    parser = argparse.ArgumentParser(description='split raw reddit/tab file.')
    parser.add_argument('--filename',help='name of file to split.')
    parser.add_argument('--start',help='optional starting line number.')
    parser.add_argument('--length', help='length of output file. (default: 500)')
    parser.add_argument('--fours', help='record sets of four', action='store_true')
    parser.add_argument('--triplets',help='record triplets', action='store_true')
    parser.add_argument('--pairs', help='record pairs', action='store_true')
    parser.add_argument('--dummy-question', help='record single dummy question')
    parser.add_argument('--mode', help='"test", "train", or "valid" - "test.big" allowed (default = "train")')
    parser.add_argument('--zip-file', help='name of zip file to archive to')
    parser.add_argument('--autoencode', help='setup files for autoencode operation. Set as percentage.')
    parser.add_argument('--stagger', help='stagger input for P.O.S.-style training.', action='store_true')
    parser.add_argument('--stagger-predict-word', help='stagger but predict only one word.', action='store_true')
    parser.add_argument('--sol', help='add eol and sol tokens.', action='store_true')
    parser.add_argument('--eol', help='add eol and sol tokens.', action='store_true')
    parser.add_argument('--xml-file', help='sentences.xml file to use.')
    parser.add_argument('--from-mnli', help='after mnli is done', action='store_true')
    parser.add_argument('--to-mnli', help='format file for later use with mnli classifier.', action='store_true')
    parser.add_argument('--from-mrpc', help='after mrpc is done', action='store_true')
    parser.add_argument('--to-mrpc', help='format file for later use with mrpc classifier.', action='store_true')
    parser.add_argument('--to-gpt2', help='format file for later use with gpt2.', action='store_true')
    parser.add_argument('--babi-for-gpt2', help='train gpt2 for training with babi synthetic data set.', action='store_true')
    parser.add_argument('--filter-possessive', help='filter only possessive sentences for gpt2.', action='store_true')
    parser.add_argument('--force', help='force normal file creation -- disable repeat detection.', action='store_true')
    parser.add_argument('--reverse', help='force reverse output.', action='store_true')


    args = parser.parse_args()
    args = vars(args)

    print(args)

    arg_filename = 'RC_2017-11'
    arg_start = 0
    arg_length = -1

    arg_end_filename = ".output.txt"

    arg_fours = False
    arg_triplets = False
    arg_pairs = False
    arg_question = ''
    arg_processed = False
    arg_zip = None #'train-files'
    arg_filelist = []
    arg_autoencode = False
    arg_stagger = False
    arg_eol = False
    arg_sol = False
    arg_xml = False

    arg_classifier = ""
    arg_to_mnli = False
    arg_from_mnli = False
    arg_to_mrpc = False
    arg_from_mrpc = False
    arg_skip_num = 8

    arg_babi_for_gpt2 = False
    arg_gpt2 = False
    arg_filter_gpt2 = False
    filter_num = 0

    arg_reverse_order = False

    arg_mode = hparams['train_name']

    arg_destination_context = ''
    arg_destination_question = ''
    arg_destination_target = ''
    arg_destination_history = ''

    if args['filename'] is not None:
        arg_filename = str(args['filename'])

    if not os.path.isfile(arg_filename):
        arg_filename = './raw/' + arg_filename
    if not os.path.isfile(arg_filename):
        print('bad path')
        exit()

    if args['start'] is not None:
        arg_start = int(args['start'])
        print('start:', arg_start)

    if args['length'] is not None:
        arg_length = int(args['length'])
        print('length:', arg_length)

    if args['triplets'] == True:
        arg_triplets = True
        arg_pairs = False
        arg_processed = True

    if args['pairs'] == True:
        arg_pairs = True
        arg_processed = True
        arg_triplets = False
        print(str(args['filename']))

    if args['dummy_question'] is not None:
        arg_question = str(args['dummy_question'])
        arg_processed = True

    if args['mode'] is not None:
        arg_mode = str(args['mode'])
        arg_processed = True
        if arg_mode != 'train' and arg_mode != 'test' and arg_mode != 'valid':
            #if arg_mode != 'train.babi' and arg_mode != 'test.babi' and arg_mode != 'valid.babi':
            if arg_mode != 'train.big' and arg_mode != 'test.big' and arg_mode != 'valid.big':
                print('bad mode')
                exit()

    if args['zip_file'] is not None:
        arg_zip = str(args['zip_file'])

    if args['autoencode'] is not None:
        arg_autoencode = float(args['autoencode'])

    if args['stagger'] == True or args['stagger_predict_word'] == True:
        arg_stagger = True

    if args['eol'] == True:
        arg_eol = True

    if args['sol'] == True:
        arg_sol = True

    if args['fours'] == True:
        arg_triplets = True
        arg_fours = True
        if not arg_stagger:
            print('not supported')
            exit()

    if args['xml_file'] is not None:
        arg_xml = True
        xml_list, xml_freq = add_to_xml(str(args['xml_file']))
        #exit()

    if args['to_mnli']:
        arg_to_mnli = True
        if arg_classifier != "":
            print('Only one classifier function at a time!')
            exit()
        arg_classifier = "MNLI"
        arg_skip_num = 8

    if args['from_mnli']:
        arg_from_mnli = True
        if arg_classifier != "":
            print('Only one classifier function at a time!')
            exit()
        arg_classifier = "MNLI"
        arg_skip_num = 8

    if args['to_mrpc']:
        arg_to_mrpc = True
        if arg_classifier != "":
            print('Only one classifier function at a time!')
            exit()
        arg_classifier = "MRPC"

    if args['from_mrpc']:
        arg_from_mrpc = True
        if arg_classifier != "":
            print('Only one classifier function at a time!')
            exit()
        arg_classifier = "MRPC"

    if args['to_gpt2'] or args['filter_possessive']:
        arg_gpt2 = True
        arg_processed = True
        arg_pairs = True
        from pytorch_pretrained_bert import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        arg_length = arg_length * 2

    if args['filter_possessive']:
        arg_filter_gpt2 = True

    if args['babi_for_gpt2']:
        arg_babi_for_gpt2 = True

    if arg_classifier != "":
        arg_end_filename = ".output.tsv"
    
    if args['force']:
        #arg_processed = False
        #arg_babi_for_gpt2 = True
        #arg_pairs = True
        #arg_eol = True # <---
        #arg_stagger = True
        #arg_question = 'eol'
        #arg_filename = os.path.abspath(arg_filename)
        pass

    if args['reverse']:
        arg_reverse_order = True

    #########

    arg_filename = os.path.abspath(arg_filename)

    arg_destination = arg_filename + arg_end_filename #'.output.txt'

    if arg_babi_for_gpt2:

        mode_b = 'babi' #arg_mode.split('.')[-1]
        filename_l = arg_filename.split('/')[-1].split('.')[0]
        if mode_b != 'babi':
            print('bad mode - input must be "babi", output must be "big"')
            exit()

        url = arg_destination.split('/')
        url = '/'.join(url[0:-1])
        print(url, filename_l, mode_b)
        print(url + '/' + filename_l + '.' + mode_b + '.from')

        z_src = open(url + '/' + filename_l + '.' + mode_b + '.from' ,'r')
        z_ques = open(url+ '/' + filename_l + '.' + mode_b + '.ques', 'r')
        z_tgt = open(url + '/' + filename_l + '.' + mode_b + '.to'  , 'r')
        file_z_src = z_src.readlines()
        file_z_ques = z_ques.readlines()
        file_z_tgt = z_tgt.readlines()
        print('open ques/tgt files')

        arg_destination_context = url + '/' + arg_mode + '.' + hparams['src_ending']
        arg_destination_target = url + '/' + arg_mode + '.' + hparams['tgt_ending']
        arg_destination_question = url + '/' + arg_mode + '.' + hparams['question_ending']

        src = open(arg_destination_context, 'w')
        ques = open(arg_destination_question, 'w')
        tgt = open(arg_destination_target, 'w')

        for i in range(len(file_z_src)):
            print(i)

            src_gpt = file_z_src[i].strip() + ' '
            ques_gpt = file_z_ques[i].strip() + '? '
            tgt_gpt = 'the ' + file_z_tgt[i].strip() + ' .'
            src.write(src_gpt + ' ')
            src.write(ques_gpt + ' ')
            src.write(' ' + tgt_gpt + '\n')

            ques.write(src_gpt + ' ')
            ques.write(ques_gpt + ' ')
            ques.write(' ' + tgt_gpt + '\n')

            tgt.write(src_gpt + ' ')
            tgt.write(ques_gpt + ' ')
            tgt.write(' ' + tgt_gpt + '\n')

        src.close()
        z_src.close()
        z_ques.close()
        z_tgt.close()

        exit()

    if not arg_processed :
        if arg_length <= 0 and arg_classifier == "":
            arg_length = 500

        ''' do split raw file '''
        lines = []
        with open(arg_filename,'r') as z:
            num = 0
            print(arg_length,'len')
            for line in z:
                if num >= arg_start and (num < arg_start + arg_length or arg_length == -1):
                    lines.append(line)
                if num > arg_start + arg_length and ( arg_length is not -1 ):
                    break
                num += 1
            z.close()


        with open(arg_destination,'w') as z:
            for line in lines:
                if arg_classifier != "MRPC" and arg_classifier != "MNLI":

                    z.write(line)
                    if not line.endswith('\n'):
                        z.write('\n')

                elif arg_to_mnli:
                    line = line.strip('\n')
                    line = line.split('\t')
                    l1 = []
                    for ii in range(arg_skip_num):
                        l1.append(" ")
                    l1.append(line[0])
                    l1.append(line[1])
                    l1.append(' ')
                    l1.append('\n')
                    l1 = '\t'.join(l1)
                    z.write(l1)

            z.close()
    else:
        ''' do split processed file '''
        if arg_length <= 0:
            arg_length = 0

        url = arg_destination.split('/')
        url = '/'.join(url[0:-1])
        print(url)
        arg_destination_context = url + '/' + arg_mode + '.' + hparams['src_ending']
        arg_destination_target = url + '/' + arg_mode + '.' + hparams['tgt_ending']
        arg_destination_question = url + '/' + arg_mode + '.' + hparams['question_ending']
        arg_destination_history = url + '/' + arg_mode + '.' + hparams['hist_ending']


        args_end_string = hparams['eol'] + ' ' + hparams['eol']
        pass

        with open(arg_filename, 'r') as z:
            num = 0

            src = open(arg_destination_context, 'w')
            tgt = open(arg_destination_target, 'w')
            arg_filelist.append(arg_destination_context.split('/')[-1])
            arg_filelist.append(arg_destination_target.split('/')[-1])

            if arg_triplets:
                ques = open(arg_destination_question, 'w')
                arg_filelist.append(arg_destination_question.split('/')[-1])

            if arg_fours:
                hist = open(arg_destination_history, 'w')
                arg_filelist.append(arg_destination_history.split('/')[-1])

            if arg_stagger:
                print('stagger output.')

            for line in z:

                ## set autoencode here.
                auto_flag = False
                if args['autoencode'] is not None and random.uniform(0, 1) < arg_autoencode: auto_flag = True
                else: auto_flag = False

                save = ''
                if num >= arg_start and (arg_length == 0 or num < arg_start + arg_length):
                    line = line.split('\t')

                    if not args['force']:
                        line[0] = format(line[0])
                        line[1] = format(line[1])

                        line[0], line[1] = move_order(line[0], line[1])

                    if arg_eol and len(line[0]) > 1:
                        line[0] =   line[0] + ' ' + hparams['eol']

                    if arg_eol and len(line[1]) > 1:
                        line[1] =   line[1] + ' ' + hparams['eol']

                    if arg_sol and len(line[0]) > 1:
                        line[0] = hparams['sol'] + ' ' +  line[0] #+ ' ' + hparams['eol']

                    if arg_sol and len(line[1]) > 1:
                        line[1] = hparams['sol'] + ' ' +  line[1] #+ ' ' + hparams['eol']

                    if arg_reverse_order == True:
                        line_temp = line[1]
                        line[1] = line[0]
                        line[0] = line_temp

                    if not arg_stagger and arg_classifier != "MRPC" and arg_classifier != "MNLI" and not arg_gpt2:

                        src.write(line[0].lower())
                        save = line[0][:]
                        if not line[0].endswith('\n'):
                            src.write('\n')
                        if arg_triplets:
                            if arg_question is not None and arg_question != '':
                                line[0] = arg_question
                            ques.write(line[0].lower())
                            if not line[0].endswith('\n'):
                                ques.write('\n')
                        if auto_flag: line[1] = save #line[0]
                        tgt.write(line[1].lower())
                        if not line[1].endswith('\n'):
                            tgt.write('\n')
                        pass

                    elif arg_gpt2:
                        if num % 2 == 0 or arg_filter_gpt2:
                            if (not arg_filter_gpt2 or
                                    sentence_contains(line[0], pronouns) or
                                    sentence_contains(line[1], pronouns)):
                                src_gpt = line[0].capitalize()
                                tgt_gpt = line[1].capitalize()
                                src.write('Q: ' + src_gpt + '\t')
                                src.write('A: ' + tgt_gpt + '\n')
                                tgt.write('Q: ' + src_gpt + '\t')
                                tgt.write('A: ' + tgt_gpt + '\n')
                                if arg_filter_gpt2:
                                    filter_num += 1
                                    print(filter_num, num)



                    elif arg_stagger:
                        src_stagger = ''
                        tgt_stagger = ''
                        ques_stagger = ''
                        hist_stagger = ''
                        src_stagger_x = ''
                        save = line[0][:]
                        save_lst = save.split(' ')
                        tgt_lst = line[1].split(' ')

                        if args['stagger_predict_word']:
                            save = line[1][:]
                            save_lst = save.split(' ')
                            src_stagger_x = line[0][:]

                        if not args['stagger_predict_word']:
                            while len(save_lst) <= len(tgt_lst):
                                save_lst.append(hparams['unk'])

                        eol_flag = False
                        for i in range(len(save_lst)):

                            word = save_lst[i]
                            next_word = ''
                            if len(save_lst) > i+1 and args['stagger_predict_word']:
                                next_word = save_lst[i + 1]

                            if i < len(tgt_lst):
                                ii = i
                            else:
                                ii = len(tgt_lst) - 1

                            ques_stagger = word
                            if len(src_stagger) > 0: src_stagger += ' '
                            src_stagger += word

                            if len(hist_stagger) > 0: hist_stagger += ' '
                            if i > 0 and tgt_stagger not in ['\n', hparams['eol'], '', '\t']:
                                hist_stagger += tgt_stagger
                            else:
                                #if i != 0: print('bad string')
                                hist_stagger = ''

                            if not args['force']:
                                src_stagger = stop_repeats(src_stagger)
                            if not args['stagger_predict_word']:
                                src.write(src_stagger.lower())
                            else:
                                src_stagger = src_stagger.replace('.', '')
                                src_stagger = src_stagger.replace(',', '')

                                src_stagger_x = src_stagger_x.replace('.', '')
                                src_stagger_x = src_stagger_x.replace(',', '')
                                src.write(src_stagger_x.lower() + ' ' + src_stagger.lower())
                            save = src_stagger
                            if not src_stagger.endswith('\n'):
                                src.write('\n')
                            if arg_triplets:
                                if arg_question is not None and arg_question != '':
                                    ques_stagger = arg_question
                                ques.write(ques_stagger.lower())
                                if not ques_stagger.endswith('\n'):
                                    ques.write('\n')
                            if arg_fours:
                                hist.write(hist_stagger.lower())
                                if not hist_stagger.endswith('\n'):
                                    hist.write('\n')
                            tgt_stagger = tgt_lst[ii]
                            if eol_flag: tgt_stagger = hparams['unk']
                            if auto_flag: tgt_stagger = word
                            if args['stagger_predict_word'] == True:
                                if next_word.endswith('.') or '.' in next_word or ',' in next_word:
                                    next_word = ' '
                                tgt_stagger = next_word
                            tgt.write(tgt_stagger.lower())
                            if tgt_stagger == hparams['eol']:
                                eol_flag = True

                            if not tgt_stagger.endswith('\n'):
                                tgt.write('\n')

                            if i != 0: num += 1

                            if arg_length != 0 and num > arg_start + arg_length:
                                print('early stop')
                                break

                        src.write(args_end_string + '\n')
                        tgt.write(hparams['eol'] + '\n')
                        if arg_triplets:
                            ques.write(args_end_string + '\n')
                        if arg_fours:
                            hist.write(args_end_string + '\n')
                        pass

                if not arg_stagger and not arg_gpt2:
                    for i in xml_freq:
                        if num % i == 0:
                            insert_xml(num, src, tgt, ques)
                            break

                if arg_length != 0 and num > arg_start + arg_length and not arg_gpt2:
                    print('early stop')
                    break

                if arg_length != 0 and (num - arg_start) > arg_start + arg_length and arg_gpt2 and not arg_filter_gpt2:
                    print('early stop -- gpt2' , num)
                    break

                if arg_length != 0 and (filter_num - arg_start) > arg_start + arg_length and arg_gpt2 and arg_filter_gpt2:
                    print('early stop -- gpt2 filter' , filter_num)
                    break

                num += 1
            src.close()
            tgt.close()
            if arg_triplets:
                ques.close()
            if arg_fours:
                hist.close()
        z.close()

        if arg_zip is not None:
            os.chdir(url)
            if len(arg_filelist) > 0:
                os.system('zip ' + arg_zip.strip() + '.zip ' + ' '.join(arg_filelist))
            pass

    print('done.')
