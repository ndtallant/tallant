'''
Zosimos was active around 300 AD and thought to be the first alchemist.
He was born in the Upper Egyptian city of Panopolis, now called Akhmim.
Zosimos allegedly wrote twenty-eight books about alchemy.
Most of what he wrote is now lost.

Anyway, this is my wrapper over sqlalchemy because I never remember
what to do and end up googling everything.
'''

import sqlalchemy
import pandas as pd

class Alchemist:
    '''A wrapper around sqlalchemy.'''

    def __init__(self, connection_string):
        '''Takes in a connection string and sets up basic things you need.'''
        self.engine = sqlalchemy.create_engine(connection_string)        
        self.schemas = sqlalchemy.inspect(self.engine).get_schema_names()
        self.tables = {s: self.engine.table_names(s) for s in self.schemas}

    def query(self, query, records=False):
        '''Returns query data as a pandas DataFrame by default. For regular
        SQLAlchemy output, use records=True.'''
        if records:
            return self.engine.execute(query).fetchall()
        return pd.read_sql(query, self.engine)


    def from_file(self, filepath, records=False):
        '''Runs a query from a SQL file.'''
        with open(filename) as f:
            sql = f.read()
        return self.query(sql, records)
