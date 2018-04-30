#!/usr/bin/python3.6

import sys
import os
sys.path.append('..')
sys.path.append(os.path.abspath('../model/'))
import model.settings as settings


hparams = settings.hparams

class Stats:
    def __init__(self):
        self.filename = hparams['save_dir'] + hparams['stats_filename'] +'.md'

    def read_stats(self):
        print(self.filename)
        pass

    def update_stats(self):
        pass

    def write_stats(self):
        pass


if __name__ == '__main__':
    s = Stats()
    s.read_stats()