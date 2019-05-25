#!/usr/bin/python3

import os
from subprocess import Popen
import re

class Commands:
    def __init__(self):
        self.erase_history = False
        self.use_async = False
        self.print_to_screen = False
        #self.enable_choose_command = False
        self.url_search = 'https://www.google.com/search?q='
        self.url_youtube = 'https://www.youtube.com/results?search_query='
        self.launch_google_chrome = 'google-chrome --app='
        #self.launch_firefox = 'firefox --search '
        self.launch_rhythmbox = 'rhythmbox '

        self.text_commands = {
            'play': 'any',
            'media': 'any',
            'google': 'search',
            'search': 'search',
            'song': 'music',
            'video': 'video',
            'movie': 'video',
            'music': 'music',
            'youtube': 'video'
        }

        self.command_dict = {
            'search': self.launch_google_chrome + self.url_search,
            'video': self.launch_google_chrome + self.url_youtube,
            'music': self.launch_rhythmbox,

        }
        self.command_string = ''

        self.p = None

    def re(self,i):
        return re.sub('[.?!:;,]','', i)

    def is_command(self,i):
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
            for xx in self.text_commands:
                if x.strip().lower() == xx.strip().lower():
                    ii.remove(x)
        return ii


    def decide_commmand(self,i):
        i = self.re(i)
        chosen = {}
        any = False
        for xx in self.text_commands.values():
            if self.print_to_screen: print(xx,'xx')
            chosen[xx] = 0
        output = False
        i = i.split()
        ii = i[:]
        if self.print_to_screen:
            print(self.text_commands)
            print(chosen)
        for x in i:
            for xx in self.text_commands:

                if x.strip().lower() == xx.strip().lower() : #and x.strip().lower() in self.text_commands:
                    output = True
                    if self.text_commands[xx] in chosen:
                        chosen[self.text_commands[xx]] += 1
                        ii.remove(x)
                        if self.print_to_screen: print(chosen[self.text_commands[xx]], xx, x)
        i = ii
        #if self.print_to_screen: print(chosen)
        if self.command_string == '':

            high = 0
            old_high = 0
            for x in chosen:
                high = chosen[x]
                if high > old_high and x != 'any':
                    self.command_string = self.command_dict[x]
                    old_high = high
                elif high > old_high and x == 'any':
                    any = True

            if self.print_to_screen: print(chosen)

            if self.command_string == '' and any is True:
                self.command_string = self.command_dict['search']

        if self.command_string != self.command_dict['music']:
            self.command_string += '+'.join(i)
        return output

    def do_command(self, i):
        erase = False
        self.command_string = ''
        if isinstance(i,list): i = ' '.join(i)
        i = self.re(i)

        #if len(self.command_string) == 0:
        self.decide_commmand(i)

        if self.print_to_screen: print(self.command_string)

        if not self.use_async:
            self.launch_sync(self.command_string)
        else:
            self.launch_async(self.command_string)

        if self.erase_history:
            erase = True
        return erase

    def launch_sync(self,i):
        os.system(i)
        pass

    def launch_async(self, i):
        i = i.split()
        self.p = Popen(i)
        pass

if __name__ == '__main__':
    c = Commands()
    command1 = 'play media'
    command2 = 'play music like video music like a movie of the allman brothers band youtube.'
    c.print_to_screen = True
    z = c.is_command(command1)
    for x in range(2):
        if len(c.strip_command(command1)) > 0:
            #command = c.strip_command(command)
            print(command1, x, 'here1')
            c.do_command(command1)
            exit()
        elif x is 1:
            #command = c.strip_command(command)
            print(command2, x, 'here2')
            c.do_command(command2)
            print('use previous command also.')
            pass
