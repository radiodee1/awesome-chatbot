#!/usr/bin/python3.6

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', help='base file.')
    parser.add_argument('--task-name', help='MNLI or MRPC')

    args = parser.parse_args()
    args = vars(args)

    print(args)

    arg_in_file = str(args['filename'])
    #arg_out_dir = '/'.join(arg_in_file.split('/')[:-2])
    #print(arg_out_dir)
    if not arg_in_file.endswith('train.tsv'):
        print('must be train file!')
        exit()
    arg_out_file = arg_in_file + '.out.tsv'
    arg_task = str(args['task_name'])

    with open(arg_in_file,'r') as read:
        with open(arg_out_file, 'w') as write:
            lines = read.readlines()
            num = 0
            skip = 0
            for line in lines:
                line = line.strip().split('\t')
                if arg_task == "MRPC":
                    out = line[3] + '\t' + line[4] + '\n'
                    if num is not 0 and float(line[0]) > 0.5:
                        write.write(out)
                    else:
                        skip += 1
                if arg_task == "MNLI":
                    out = line[8] + '\t' + line[9] + '\n'
                    if str(line[11]) == "entailment" or str(line[11]) == "neutral" : #> 0.3:
                        write.write(out)
                    else:
                        skip += 1
                num += 1
            print(num,'number', skip, "skip")