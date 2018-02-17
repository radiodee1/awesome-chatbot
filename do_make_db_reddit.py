#!/usr/bin/python3

import sqlite3
import json
from datetime import datetime
import os
import sys
import itertools

timeframe = 'raw/RC_2015-02'
dbname = 'input'
sql_transaction = []

add_simple_question = False
newlinechar = ' '
#newlinechar = ' newlinechar '

connection = sqlite3.connect('{}.db'.format(dbname))
c = connection.cursor()

def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT, unix INT, score INT)")

def format_data(data):
    data = data.replace('\n', newlinechar ).replace('\r', newlinechar ).replace('"',"'")
    return data

def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except:
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

def sql_insert_complete(commentid,parentid,parent,comment,subreddit,time):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id,parent, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(parentid, commentid,parent, comment, subreddit, int(time), 5)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))

def acceptable(data):
    if len(data.split(' ')) > 50 or len(data) < 1:
        return False
    elif len(data) > 1000:
        return False
    elif data == '[deleted]':
        return False
    elif data == '[removed]':
        return False
    else:
        return True

def find_parent(pid):
    try:
        sql = "SELECT comment FROM parent_reply WHERE comment_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else: return False
    except Exception as e:
        #print(str(e))
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
        #print(str(e))
        return False
    
if __name__ == '__main__':


    print(sys.argv)

    if len(sys.argv) > 1:
        timeframe = sys.argv[1]
        print(timeframe)
        print('this first arg should be the path to the reddit json dump file.')

    create_table()
    row_counter = 0
    start = 0
    paired_rows = 0
    xx = 16

    if len(sys.argv) > 2:
        row_counter = int(sys.argv[2])
        start = row_counter
        print(start)
        print('this second arg is typically an integer val with five zeros.')


    with open('{}'.format(timeframe), buffering=1000) as f:
        #for row in f:
        for row in itertools.islice(f, start, None):
            row_counter += 1
            row = json.loads(row)
            parent_id = row['parent_id']
            body = format_data(row['body'])
            created_utc = row['created_utc']
            score = row['score']
            try:
                score = int(row['score'])
            except:
                score = 0

            try:
                comment_id = row['name']
            except:
                comment_id = 't1_' + row['id']

            #comment_id = row['name']

            subreddit = row['subreddit']
            parent_data = find_parent(parent_id)

            if add_simple_question and paired_rows % xx == 0 and paired_rows > xx:
                ## auto-encoder type question. ##
                text = "i am {} . who is this ? it's me .".format(subreddit)
                sql_insert_complete(comment_id + '_z', parent_id, text, text, subreddit, created_utc)
                paired_rows += 1
                pass

            elif add_simple_question and paired_rows % xx == 1 and paired_rows > xx:
                ## auto-encoder type question. ##
                sql_insert_complete(comment_id + '_z', parent_id, body, body, subreddit, created_utc)
                paired_rows += 1
                pass

            elif int(score) >= 2:
                existing_comment_score = find_existing_score(parent_id)
                if existing_comment_score:
                    if score > existing_comment_score:
                        if acceptable(body):
                            sql_insert_replace_comment(comment_id,parent_id,parent_data,body,subreddit,created_utc,score)
                            
                else:
                    if acceptable(body):
                        if parent_data:
                            sql_insert_has_parent(comment_id,parent_id,parent_data,body,subreddit,created_utc,score)
                            paired_rows += 1
                        else:
                            sql_insert_no_parent(comment_id,parent_id,body,subreddit,created_utc,score)
                            
            if row_counter % 100000 == 0:
                print('Total Rows Read: {}, Paired Rows: {}, Time: {}'.format(row_counter, paired_rows, str(datetime.now())))
                
