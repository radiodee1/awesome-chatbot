#!/usr/bin/python3

import os
import sys
from subprocess import Popen
import re
import xml.etree.ElementTree as ET
sys.path.append('..')
from model.settings import hparams as hp

aiml_txt = hp['data_dir'] + '/commands.xml'

class Commands:
    def __init__(self):
        self.words_dupe = []
        self.text_pattern = []
        self.text_template = []
        self.text_separate = []
        self.command_string = ''
        self.text_commands = []
        self.p = None
        self.print_to_screen = False
        self.erase_history = True
        self.use_async = False
        self.sep_list = ['google-chrome', 'firefox']

        self.url_search = 'https://www.google.com/search?q='
        self.url_youtube = 'https://www.youtube.com/results?search_query='
        self.launch_google_chrome = 'google-chrome --app='
        self.launch_firefox = 'firefox --search '
        self.launch_rhythmbox = 'rhythmbox '
        self.launch_mail = 'thunderbird'
        self.launch_office = 'libreoffice'
        self.launch_file = 'nautilus'
        self.launch_terminal = 'gnome-terminal'

        self.command_dict = {
            'search': self.launch_google_chrome + self.url_search,
            'video': self.launch_google_chrome + self.url_youtube,
            'music': self.launch_rhythmbox,
            'mail': self.launch_mail,
            'office': self.launch_office,
            'file': self.launch_file,
            'terminal': self.launch_terminal,
            'firefox': self.launch_firefox
        }

        self.setup_for_aiml()

    def setup_for_aiml(self):
        tree = ET.parse(aiml_txt)
        root = tree.getroot()
        self.text_pattern = []
        self.text_template = []

        for x in root:
            pattern = ''
            template = ''
            for y in x:
                if y.tag == 'pattern':
                    pattern = y.text.strip()
                if y.tag == 'template':
                    template = y.text.strip()
            #print(pattern, template)

            self.text_pattern.append(pattern)
            self.text_template.append(template)

        temp_pattern = []
        for x in self.text_pattern:
            x = self.re(x).lower()
            temp_pattern.append(x.split(' '))

        if self.print_to_screen: print(temp_pattern,'\n---------')

        ## remove dupes ##
        dupe_list = []
        for i in range(len(temp_pattern)):
            ii = temp_pattern[i]
            for j in range(len(ii)):
                jj = ii[j]
                for k in range(len(temp_pattern)):
                    if jj in temp_pattern[k] and k != i:
                        if jj not in dupe_list:
                            dupe_list.append(jj)

        if self.print_to_screen: print(dupe_list)
        self.words_dupe = dupe_list

        for i in range(len(temp_pattern)):
            for j in range(len(temp_pattern[i]) ,0,-1):
                j -= 1
                #print(j)
                x = temp_pattern[i][j]
                if x in dupe_list:
                    del(temp_pattern[i][j])

        if self.print_to_screen: print(temp_pattern)
        self.text_separate = temp_pattern

        for i in self.text_separate:
            for j in i:
                self.text_commands.append(j)

        if self.print_to_screen: print(self.text_commands)

        if self.print_to_screen: print(self.text_template, '<<<')

    def re(self,i):
        return re.sub('[.?!:;,]','', i)

    def is_command(self,i):
        if not isinstance(i, str): i = ''
        i = self.re(i)
        output = False
        for x in i.split():
            for xx in self.text_commands:
                if x.strip().lower() == xx.strip().lower():
                    output = True
        return output

    def strip_command(self,i):
        i = self.re(i)
        i = i.split()
        ii = i[:]
        for x in i:
            for xx in self.words_dupe:
                if x.strip().lower() == xx.strip().lower():
                    ii.remove(x)
        return ii


    def decide_commmand(self, i, strip_contents=True):
        self.command_string = ''

        ## if pattern matches exactly ##
        for j in range(len(self.text_pattern)):
            if i.lower().strip().startswith(self.text_pattern[j].lower().strip()):
                self.command_string = self.text_template[j]
                if self.text_template[j].strip() in self.command_dict:
                    self.command_string = self.command_dict[self.text_template[j]]
                    #print(self.command_string,'cs')
                else:
                    self.command_string = self.text_template[j]
                ## decide how to end command ##
                ii = self.command_string
                tp = i[len(self.text_pattern[j]):]
                tp = tp.strip().split(' ')
                separator = ' '
                space = ' '
                add_txt = False
                for zz in self.sep_list:
                    if self.print_to_screen: print(zz, ii, 'zz,ii')
                    if zz in ii :
                        separator = '+'
                        space = ''
                        add_txt = True
                        if self.print_to_screen: print(zz,separator)

                if add_txt:
                    self.command_string += space + separator.join(tp)

        if self.print_to_screen: print(self.command_string,'<--')

        ## match by occurance of special words ##
        if self.command_string == '' and self.is_command(i):
            if strip_contents:
                i = self.strip_command(i)
            if self.print_to_screen: print(i)
            chosen = {}
            commands = {}
            for j in range(len(self.text_separate)):
                for jj in self.text_separate[j]:
                    chosen[jj] = j
                    commands[j] = 0

            for j in i:
                for k in chosen:
                    if j.lower().strip() == k.lower().strip():
                        tot = commands[chosen[k]]
                        tot += 1
                        commands[chosen[k]] = tot

            if self.print_to_screen: print(chosen, commands)

            highest = 0
            highest_index = 0
            for j in commands:
                if int(commands[j]) > highest:
                    highest = commands[j]
                    highest_index = j

            ## decide how to end command ##
            if self.text_template[highest_index].strip() in self.command_dict:
                self.command_string = self.command_dict[self.text_template[highest_index].strip()]
            else:
                self.command_string = self.text_template[highest_index]
            separator = ' '
            space = ' '
            add_txt = False
            for zz in self.sep_list:
                #print(zz, 'zz')
                if zz in self.command_string:
                    separator = '+'
                    space = ''
                    add_txt = True
                    #print(zz,'zz')
            #self.command_string = self.text_template[highest_index].strip()
            if add_txt:
                self.command_string += space + separator.join(i)
                if self.print_to_screen: print(i, 'i')

            if self.print_to_screen: print(self.command_string,'<==')
        pass

    def do_command(self, i, strip_contents=True):
        erase = False
        self.command_string = ''
        if isinstance(i,list): i = ' '.join(i)
        if strip_contents:
            i = self.re(i)

        self.decide_commmand(i, strip_contents)

        if self.print_to_screen or True: print(self.command_string)

        if not self.use_async:
            self.launch_sync(self.command_string)
        else:
            self.launch_async(self.command_string)

        if self.erase_history:
            erase = True
        return erase

    def launch_sync(self,i):
        ## if the program doesn't exist, this command will fail but chatbot will continue.
        os.system(i)
        pass

    def launch_async(self, i):
        i = i.split()
        self.p = Popen(i)
        pass

if __name__ == '__main__':
    c = Commands()


    command1 = 'movies allman brothers band'
    command2 = 'play music like video music like a movie of the music band youtube.'
    command3 = 'hello there'
    command4 = 'find allman brothers band'
    c.print_to_screen = True
    c.setup_for_aiml() ## this gets executed twice
    print('command 1')
    z = c.decide_commmand(command1)
    print('command 3')
    z = c.decide_commmand(command3)
    print('command 4')
    z = c.decide_commmand(command4)

    exit()
