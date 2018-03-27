#!/usr/bin/python3

import codecs
import sqlite3
import os
from datetime import datetime
import sys

print(sys.argv)

txtname = 'raw/eng-fra.txt'

if len(sys.argv) > 1:
    txtname = sys.argv[1]
    print(txtname)
    print('this first arg should be the path to the french corpus file.')

timeframe = 'input'
sql_transaction = []

shift_and_repeat = False
test_on_screen = True

connection = sqlite3.connect('{}.db'.format(timeframe))
c = connection.cursor()

def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT, unix INT, score INT)")

def format_data(data):
    #data = str(data)

    data2 = data.replace('\n',' newlinechar ').replace('\r',' newlinechar ').replace('"',"'")
    #data = data[:]
    #data = data.encode('utf8')
    return data2

def transaction_bldr(sql , force=False):
    global sql_transaction
    if not force: sql_transaction.append(sql)
    if len(sql_transaction) > 1000 or force:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except Exception as e:
                print(str(e))
                exit()
                pass
        connection.commit()
        sql_transaction = []

def sql_insert_replace_comment(commentid,parentid,parent,comment,subreddit,time,score):
    try:
        sql = """UPDATE parent_reply SET parent_id = ?, comment_id = ?, parent = ?, comment = ?, subreddit = ?, unix = ?, score = ? WHERE parent_id =?;""".format(parentid, commentid, parent, comment, subreddit, int(time), score, parentid)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))

def sql_insert_has_parent(commentid,parentid,parent,comment,subreddit,time,score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(parentid, commentid, parent, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))

def sql_insert_no_parent(commentid,parentid,comment,subreddit,time,score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}",{},{});""".format(parentid, commentid, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))

def sql_insert_complete(commentid,parentid,parent,comment,subreddit,time,score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id,parent, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(parentid, commentid,parent, comment, subreddit, int(time), 5)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))
        exit()


def acceptable(data):
    return True
    '''
    if len(data.split(' ')) > 500 or len(data) < 1:
        return False
    elif len(data) > 1000:
        return False
    elif data == '[deleted]':
        return False
    elif data == '[removed]':
        return False
    else:
        return True
    '''

def find_parent(pid):
    try:
        sql = "SELECT comment FROM parent_reply WHERE comment_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else: return False
    except Exception as e:
        print(str(e))
        return False

def find_existing_score(pid):
    try:
        sql = "SELECT score FROM parent_reply WHERE parent_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else: return False
    except Exception as e:
        print(str(e))
        return False
    
if __name__ == '__main__':
    create_table()
    row_counter = 0
    paired_rows = 0
    #txtname = 'movie_lines'
    with codecs.open('{}'.format(txtname), 'r' ,buffering=1000) as z:
        f = z.readlines()
        bucket = ''
        row = ''
        rownext = ''
        row_out = ''
        num = 0
        body = ''
        reply = ''
        name = ''
        done = False
        done_counter = 0
        comment_id = ''
        parent_id = ''
        comment_id_name = ''
        
        for j in range(len(f)): 
        
            #print(f[j],'read')

            num = j
            comment_id = 'name-'+str(num)
            parent_id = 'parent-'+ str(num+1)
            #comment_id_name = comment_id + ' ' + str(num)

            created_utc = num #'utc_'+ str(num)
            score = 5  

            subreddit = 0  

            reply = str(f[j])

            r = reply.strip('\n').split('\t')
            print(r,'r')
            if len(r) > 1:
                reply = r[0]
                body = r[1]

                paired_rows += 1
                row_counter += 1
                sql_insert_complete(comment_id,parent_id,body,reply,subreddit,created_utc,score)


            if row_counter % 100000 == 0:
                print('Total Rows Read: {}, Paired Rows: {}, Time: {}'.format(row_counter, paired_rows, str(datetime.now())))

            #if done:


    transaction_bldr('', force=True)

    os.system("mv input.db raw/input_french.db")
