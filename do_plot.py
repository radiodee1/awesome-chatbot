#!/usr/bin/python3.6

from __future__ import unicode_literals, print_function, division

import sys
sys.path.append('..')
from io import open
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import json
from model.settings import hparams
import glob

if __name__ == '__main__':
    #os.chdir('/' + '/'.join(sys.argv[0].split('/')[:-1]))
    parser = argparse.ArgumentParser(description='Plot some NMT values.')
    parser.add_argument('--files', help='File glob for plotting. Must be json files!!')
    parser.add_argument('--title', help='Graph title.')

    args = parser.parse_args()
    args = vars(args)
    print(args)

    do_filelist = False
    do_title_graph = False

    if args['files'] is not None:
        do_filelist = True

    if args['title'] is not None:
        do_title_graph = True

    arg_filename = '/'.join( hparams['save_dir'].split('/')[1:]) + '/' + 'test*.json'
    arg_title = 'Autoencode Accuracy vs. Steps'

    if do_filelist:
        arg_filename = str(args['files'])

    if do_title_graph:
        arg_title = str(args['title'])

    arg_filelist = arg_filename.split(',')
    arg_glob_list = []
    for i in arg_filelist:
        print(i,'use for plot')
        arg_glob_list.extend(glob.glob(i))

    print(arg_glob_list)
    arg_list = []
    for i in arg_glob_list:
        if os.path.isfile(i):
            with open(i, 'r') as z:
                sublist = []
                j = json.loads(z.read())
                for k in j:
                    sublist.append((int(k), float(j[k])))
                sublist.sort(key=lambda tuple: tuple[0])
                #print(sublist)
                arg_list.append(sublist)

    #print(arg_list)

    arg_plot_color = [ 'r', 'b', 'g', 'y','c','m']
    fig, ax = plt.subplots()
    plt.ylabel('Accuracy')
    plt.xlabel('Sentence Pairs')
    plt.title(arg_title)
    handles = []
    for i in range(len(arg_list)):
        ii = i % len(arg_plot_color)
        label_out = arg_glob_list[i].split('/')[-1]
        color_patch = mpatches.Patch(color=arg_plot_color[ii], label=label_out)
        handles.append(color_patch)

        lst_x = []
        lst_y = []
        for k in arg_list[i]:
            lst_x.append(k[0])
            lst_y.append(k[1])
            ax.plot(lst_x, lst_y, arg_plot_color[ii] + '-')
    ax.legend(handles=handles)
    plt.show()
