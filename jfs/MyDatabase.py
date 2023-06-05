import os
import socket

from dbutils.pooled_db import PooledDB
import pymysql
import configparser
import json


class MyDatabase:
    def __init__(self, conf):
        self.conf = conf

    def connect(self):
        parser = configparser.ConfigParser()
        parser.read(self.conf, encoding="utf-8")
        db_config = dict(parser['database'])
        # 创建连接池
        pool = PooledDB(pymysql,
                        maxconnections=5,
                        mincached=2,
                        maxcached=5,
                        host=db_config['host'],
                        port=int(db_config.get('port', '3306')),
                        user=db_config.get('user', 'root'),
                        password=db_config['password'],
                        database=db_config['database'])

        conn = pool.connection()
        return conn

    def fetch(self, name):
        sql = 'SELECT chunkid, files, name FROM jfs_chunk_file WHERE name LIKE %s'
        with self.connect().cursor() as cursor:
            cursor.execute(sql, (f"{name}%"))
            result = cursor.fetchall()
            cursor.connection.commit()
            return result

    def query_mount(self):
        sql = "SELECT info FROM jfs_session2"
        with self.connect().cursor() as cursor:
            cursor.execute(sql)
            session2 = cursor.fetchall()

            hostname = socket.gethostname()

            def parse_json(js):
                return json.loads(js.decode("utf-8"))

            mount = ""
            for row in session2:
                info = parse_json(row[0])
                if hostname == info["HostName"] and os.path.sep in info["MountPoint"]:
                    mount = info["MountPoint"]
                    break

            return mount
