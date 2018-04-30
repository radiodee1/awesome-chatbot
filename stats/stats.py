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
        self.num_of_tests = 20
        self.heading = []
        self.body = []

    def read_stats(self):
        found_heading = False
        found_divider = False
        print(self.filename)
        if os.path.isfile(self.filename):
            with open(self.filename, 'r') as z:
                zz = z.readlines()
                for i in range(len(zz)):
                    if found_heading is False:
                        line = zz[i].split('|')
                        found_divider = False
                        found_heading = True
                        for l in line:
                            l = l.strip()
                            if len(l) > 0:
                                self.heading.append(l)
                            else:
                                self.heading.append('blank')
                    elif found_heading is True and found_divider is False:
                        ''' consume divider '''
                        pass
                    else:
                        line = zz[i].split('|')
                        data = []
                        for l in line:
                            l = l.strip()
                            if len(l) > 0:
                                data.append(l)
                            else:
                                data.append('0')
                        self.body.append(data)

        pass

    def update_stats(self):
        pass

    def write_stats(self):
        pass

    def print_stats(self):
        print(self.heading)
        print(self.body)


if __name__ == '__main__':
    s = Stats()
    s.read_stats()
    s.print_stats()