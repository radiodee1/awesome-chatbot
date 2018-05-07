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
        self.num_of_tests = 20 + 2
        self.heading = []
        self.body = []
        self.score = 555
        self.test = '5'
        self.column_name = 'small_embedding'
        self.skip_new_score = True
        self.skip_row_number = True
        self.has_labeled_row_number = True
        self.table_first_col = [' ']
        self.table_out = []
        self.text_before = []
        self.text_after = []
        self.b = None
        print('try:')
        print('python3.6 stats.py --load-babi --basename=babi --conserve-space --babi-num=1')
        if len(sys.argv) == 1: exit()

    def read_stats(self):
        found_text_before = False
        found_text_after = False
        found_heading = False
        found_divider = False
        print(self.filename)
        if os.path.isfile(self.filename):
            with open(self.filename, 'r') as z:
                zz = z.readlines()
                #print(len(zz))
                for i in range(len(zz)):
                    if '|' not in zz[i] and found_text_before == False:
                        self.text_before.append(zz[i])
                        continue
                    if '|' not in zz[i] and found_text_before:
                        self.text_after.append(zz[i])
                        continue
                    if found_heading is False:
                        line = zz[i].strip('\n').split('|')
                        found_text_before = True
                        found_divider = False
                        found_heading = True
                        for l in range(len(line)):
                            ll = line[l].strip()
                            if len(ll) > 0:
                                self.heading.append(ll)
                            elif l != 0 and l != len(line) -1 :
                                #self.heading.append('blank')
                                pass
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
                            if self.skip_row_number and l == 1 and not self.has_labeled_row_number:
                                data.append(ll)
                                continue
                            elif l == 1 and self.has_labeled_row_number:
                                self.table_first_col.append(ll)
                                continue
                            if len(ll) > 0 :
                                data.append(ll)
                            elif l != 0 and l != len(line) -1:
                                data.append('0')
                        self.body.append(data)

        pass

    def update_stats(self):
        if not self.skip_new_score:
            self.get_score()
        table = []

        if self.column_name not in self.heading: self.heading.append(self.column_name)
        for j in range(self.num_of_tests + 1):
            row = []
            if j == 0:
                self.heading = [' '] + self.heading
                table.append(self.heading)
            if j != 0:
                if not self.has_labeled_row_number:
                    row.append(str(j))
                elif len(self.table_first_col) > j:
                    row.append(self.table_first_col[j])

                for i in range(0,len(self.heading) -1):
                    if self.heading[i+1].strip() == self.column_name.strip() and str(j) == str(self.test):
                        row.append(str(self.score))
                    elif j -1 < len(self.body) and i < len(self.body[j-1]):
                        row.append(self.body[j-1][i])
                    else:
                        row.append('0')
                table.append(row)
        self.table_out = table
        print(self.text_before)
        print(table)
        print(self.text_after)

    def get_score(self):
        import model.babi_ii as babi

        b = babi.NMT()
        print(b.printable,'print')
        self.column_name = b.printable
        b.setup_for_babi_test()
        print(b.score,'score!!')
        self.test = b.babi_num
        self.score = '%.2f' % b.score
        print(self.test,'num')
        pass

    def write_stats(self):
        with open(self.filename+'.md','w') as z:
            for i in self.text_before:
                z.write(i)
            for i in range(len(self.table_out)):
                if i == 0:
                    l = ' | ' + ' | '.join(self.table_out[i]) + ' | ' + '\n'
                    z.write(l)
                    l = ''
                    for _ in range(len(self.table_out[i])):
                        l = l + '-|'
                    l = '|' + l + ' \n'
                    z.write(l)
                else:

                    l = ' | ' + ' | '.join(self.table_out[i]) + ' | ' + '\n'
                    z.write(l)
            for i in self.text_after:
                z.write(i)
            z.close()
        pass

    def print_stats(self):
        print(self.heading)
        print(self.body)
        print(self.table_first_col)


if __name__ == '__main__':
    s = Stats()
    s.read_stats()
    s.update_stats()
    s.write_stats()
    s.print_stats()