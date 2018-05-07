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
        self.dbname = dbname
        self.dbhost = dbhost
        self.dbport = dbport
        self.dbusername = dbusername
        self.dbpasswd = dbpasswd
        
        if override:
            #logger.info("Overriding DB connection params")
            self.dbname=DBVars.dbname
            self.dbhost=DBVars.dbhost
            self.dbport=DBVars.dbport
            self.dbusername=DBVars.dbusername
            self.dbpasswd=DBVars.dbpasswd

        self.conn = self.openConnection() 
        print("Don't forget to close!")
    
    def openConnection(self):
        '''Opens a connection to a psql database, using self.db params'''
        #logger.debug("Opening a Connection")
        conn = psycopg2.connect(dbname=self.dbname, 
                                user=self.dbusername, 
                                password=self.dbpasswd, 
                                host=self.dbhost,
                                port=self.dbport)
        return conn 

    def closeConnection(self):
        '''Closes any active connection'''
        #logger.debug("Closing Connection")
        self.conn.close() 
        return True

    def createTables(self):
        '''
        Creates a trip and station table in the database. 
        Commits those changes to the database.
        Creates and closes a new cursor each time it is run.
        Returns True on success. 
        '''
        #logger.debug("Creating Tables")
        try: 
            cur = self.conn.cursor() 
            #make a table 
            self.conn.commit()
            cur.close()
        
        except NameError: 
            print('There is no connection for this client')
        
        return True 
