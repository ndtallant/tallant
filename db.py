'''
db.py

Author: Nick Tallant

This file contains clients for different database engines, currently:
    postgres
    sqlite3 in near future
'''

import psycopg2

class Elephant:
    '''
    psql client
    '''
    def __init__(self, dbname, dbhost, dbport, dbusername, dbpasswd, override=False):
    # you add class variables here like
    # self.myvar="the greatest variable ever. the best"
    self.dbname = dbname
    self.dbhost = dbhost
    self.dbport = dbport
    self.dbusername = dbusername 
    self.dbpasswd = dbpasswd 

    #for grading do not modify
    if override:
        logger.info("Overriding DB connection params")
        self.dbname=DBVars.dbname
        self.dbhost=DBVars.dbhost
        self.dbport=DBVars.dbport
        self.dbusername=DBVars.dbusername
        self.dbpasswd=DBVars.dbpasswd

    pass
