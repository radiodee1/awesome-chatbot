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
        self.score = 0
        self.test = 1
        self.column_name = ''
        self.skip_new_score = False
        #if len(sys.argv) > 1:
        #    self.test = int(sys.argv[1])

    def read_stats(self):
        found_heading = False
        found_divider = False
        print(self.filename)
        if os.path.isfile(self.filename):
            with open(self.filename, 'r') as z:
                zz = z.readlines()
                #print(len(zz))
                for i in range(len(zz)):
                    if found_heading is False:
                        line = zz[i].strip('\n').split('|')

                        found_divider = False
                        found_heading = True
                        for l in range(len(line)):
                            ll = line[l].strip()
                            if len(ll) > 0:
                                self.heading.append(ll)
                            elif l != 0 and l != len(line) -1 :
                                self.heading.append('blank')
                    elif found_heading is True and found_divider is False:
                        ''' consume divider '''
                        found_divider = True
                        pass
                    else:
                        #print(i,'i')
                        line = zz[i].strip('\n').split('|')
                        data = []
                        for l in range(len(line)):
                            ll = line[l].strip()
                            if len(ll) > 0:
                                data.append(ll)
                            elif l != 0 and l != len(line) -1:
                                data.append('0')
                        self.body.append(data)

        pass

    def update_stats(self):
        if not self.skip_new_score:
            self.get_score()


    def get_score(self):
        import model.babi_ii as babi

        b = babi.NMT()
        self.column_name = b.printable
        b.setup_for_babi_test()
        print(b.score,'score!!')
        self.test = b.babi_num
        self.score = b.score
        print(self.test,'num')
        pass

    def write_stats(self):
        pass

    def print_stats(self):
        print(self.heading)
        print(self.body)


if __name__ == '__main__':
    s = Stats()
    s.read_stats()
    s.update_stats()
    s.print_stats()