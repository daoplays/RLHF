import sqlite3
from sqlite3 import Error
from log import *
import numpy as np

# setup the connection to the database and create the table if required
def create_database_connection(db_file = r"preferences.db"):
    """ create a database connection to the SQLite database
    specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
	
    conn = None
    try:
        conn = sqlite3.connect(db_file, isolation_level=None)
		
    except Error as e:
        log_error(str(e))
        return None
		
    success = create_table(conn)

    if (not success):
        return None

    return conn
	
# create the table with the structure required:
# primary key is just index into the table
# each segment contains obs, next obs and action
# we include 2 segments per row, and the distribution of the preference ([1, 0], [0, 1], or [0.1, 0.5])
def create_table(conn):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :return:
    """
	
    create_preferences_table = """ CREATE TABLE IF NOT EXISTS preferences (
        id integer PRIMARY_KEY,
        obs_1x1 integer NOT NULL,
        obs_1x2 integer NOT NULL,
        action_1 integer NOT NULL,
        obs_2x1 integer NOT NULL,
        obs_2x2 integer NOT NULL,
        action_2 integer NOT NULL,
        mu_1 real NOT NULL,
        mu_2 real NOT NULL); """
		
    try:
        c = conn.cursor()
        c.execute(create_preferences_table)
        c.close()
    except Error as e:
        log_error(str(e))
        return False

    return True
		
# inset a set of rows into the table within a single transaction
def insert_rows(conn, rows):
    """
    Create a new entry in the preferences table
    :param conn:
    :param row:
    :return: project id
    """
    sql = ''' INSERT INTO preferences(id,obs_1x1,obs_1x2,action_1,obs_2x1,obs_2x2,action_2,mu_1,mu_2)
	      VALUES(?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute("begin")
    for row in rows:
        cur.execute(sql, row)
    cur.execute("commit")
    cur.close()

def get_rows(conn):
    sql = ''' SELECT * FROM preferences '''
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()

    return np.array(rows)