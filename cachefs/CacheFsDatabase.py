import os
import socket
import pymysql
import configparser
import json
from dbutils.pooled_db import PooledDB


class CacheFsDatabase:
    def __init__(self, conf):
        self.conf = conf
        self.pool = self.connection_pool()

    def connection_pool(self):
        """
        The connection_pool function creates a connection pool to the database.
            The function takes in no arguments and returns a connection pool object.
        :param self: Represent the instance of the class
        :return: A pooleddb object
        :doc-author: Trelent
        """
        parser = configparser.ConfigParser()
        parser.read(self.conf, encoding="utf-8")
        db_config = dict(parser['database'])
        # create mysql connect pool
        return PooledDB(pymysql,
                        maxconnections=1000000,
                        mincached=2,
                        maxcached=1000000,
                        host=db_config['host'],
                        port=int(db_config.get('port', 3306)),
                        user=db_config.get('user', 'root'),
                        password=db_config['password'],
                        database=db_config['database'])

        # conn = pool.connection()
        # return conn


    def fetch(self, name, sql):
        """
        The fetch function takes in a name and sql statement.
            The function then uses the connection pool to connect to the database,
            execute the query with the given name as a parameter, and return all results.

        :param self: Represent the instance of the class
        :param name: Search for the file name in the database
        :param sql: Pass in the sql statement to be executed
        :return: A list of tuples, each tuple is a row from the database
        :doc-author: Trelent
        """
        # sql = 'SELECT b.chunkid, b.files, a.name FROM jfs_chunk_file a INNER JOIN jfs_slice_file b ON a.chunkid = b.chunkid WHERE a.name LIKE %s'  #'SELECT chunkid, files, name FROM jfs_chunk_file WHERE name LIKE %s'
        with self.pool.connection() as conn, conn.cursor() as cursor:
                cursor.execute(sql, (f"{name}%"))
                result = cursor.fetchall()
                # cursor.connection.commit()
                return result


    def query_slices(self, chunk_ids):
        """
        The query_slices function takes a list of chunk_ids and returns the files associated with each chunk.

        :param self: Represent the instance of the class
        :param chunk_ids: Specify the chunk ids of the chunks we want to retrieve
        :return: The chunkid and the files
        :doc-author: Trelent
        """
        sql = 'SELECT chunkid, files FROM jfs_slice_file WHERE chunkid IN (%s)'
        placeholders = ','.join(['%s'] * len(chunk_ids))
        final_query = sql % placeholders
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute(final_query, chunk_ids)
            return cursor.fetchall()


    def query_mount(self):
        """
        The query_mount function is used to find the mount point of the JFS.
        It does this by querying the jfs_session2 table in MySQL for a row that contains
        the hostname of this machine and a MountPoint value that includes os.path.sep, which
        is '/' on Linux systems.

        :param self: Represent the instance of the class
        :return: The mount point of the session
        :doc-author: Trelent
        """
        sql = "SELECT info FROM jfs_session2"
        with self.pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql)
            session2 = cursor.fetchall()
            hostname = socket.gethostname()
            mount = ""
            for row in session2:
                try:
                    info = json.loads(row[0].decode("utf-8"))
                    if hostname == info["HostName"] and os.path.sep in info["MountPoint"]:
                        mount = info["MountPoint"]
                        break
                except (json.JSONDecodeError, KeyError):
                    pass
            return mount
