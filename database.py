'''
db.py

Author: Nick Tallant

This file contains clients for different database engines, currently:
    postgres
    sqlite3

It may be cleaner to have one client class, and have the db-engine specified.
But will have to take a look at the different APIs.
'''
import psycopg2
import sqlite3
import pandas as pd

class Elephant:
    '''
    psql client
    '''
    def __init__(self, dbname='', dbhost='', dbusername='', dbpasswd=''):
        self.dbname = dbname
        self.dbhost = dbhost
        #self.dbport = dbport
        self.dbusername = dbusername
        self.dbpasswd = dbpasswd
        self.conn = self.open_connection() 
    
    def open_connection(self):
        '''Opens a connection to a psql database, using self.db params'''
        try: 
            conn = psycopg2.connect(dbname=self.dbname, 
                                    user=self.dbusername, 
                                    password=self.dbpasswd, 
                                    host=self.dbhost)
            
            print('Connected to', self.dbname)
            return conn 
        
        except ConnectionError:
            print('Can\'t connect to the database!')

    def close_connection(self):
        '''Closes any active connection'''
        self.conn.close() 
        print('Connection closed') 
        return True

    def basic_query(self, query):
        cur = self.conn.cursor()
        cur.execute(query)
        data = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        cur.close()
        return pd.DataFrame.from_records(data, columns=cols)
    
    def create_tables(self):
        '''
        Creates a trip and station table in the database. 
        Commits those changes to the database.
        Creates and closes a new cursor each time it is run.
        Returns True on success. 
        '''
        try: 
            cur = self.conn.cursor() 
            #make a table 
            self.conn.commit()
            cur.close()
        
        except NameError: 
            print('There is no connection for this client')
        
        return True


class Feather:
    '''
    SQLite client
    '''
    def __init__(self, dbname=''):
        self.dbname = dbname
        self.conn = self.open_connection() 

    def open_connection(self):
        '''Opens a connection to a psql database, using self.db params'''
        conn = sqlite3.connect(self.dbname) 
        print('Connected to', self.dbname) 
        return conn 

    def close_connection(self):
        '''Closes any active connection'''
        self.conn.close() 
        print('Closed connection') 
        return True

    def basic_query(self, query):
        cur = self.conn.cursor()
        cur.execute(query)
        data = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        cur.close()
        return pd.DataFrame.from_records(data, columns=cols)

    def __enter__(self):
        return self 
    
    def __exit__(self, *args):
        '''Guarantees a closed connection, *args are three exception types.'''        
        self.close_connection()
