import configparser
import json
import random
import tarfile
from itertools import chain
from queue import Queue
from dbutils.pooled_db import PooledDB
import pymysql
from numpy import array2string



# 从配置文件中读取配置信息
parser = configparser.ConfigParser()
parser.read('/home/wjy/db.conf', encoding="utf-8")
db_config = dict(parser['database'])
 # 设置连接池参数
pool = PooledDB(
    creator=pymysql,
    maxconnections=5,
    mincached=2,
    maxcached=5,
    blocking=True,
    host=db_config['host'],
    port=int(db_config['port']),
    user=db_config['user'],
    password=db_config['password'],
    database=db_config['database'],
    # charset=db_config['charset']
)

conn = pool.connection() # 获取连接
# 使用连接进行数据库操作
cursor = conn.cursor()
names="imagenet_4M"
sql='SELECT chunkid, files, name FROM jfs_chunk_file WHERE name LIKE %s"%%"'
cursor.execute(sql, names)
result = cursor.fetchall()
print(result)

cursor.execute("SELECT info FROM jfs_session2 ")
session2 = cursor.fetchone()
print(session2)
cursor.close() # 关闭游标
conn.close() # 关闭连接

for r in session2:
    js = json.loads(r.decode("utf-8"))
    print(js["MountPoint"])

mount = js["MountPoint"]
 # 将查询结果放入map中
chunkIds = []
maps = {}
fileMaps = {}
for row in result:
    chunkIds.append(row[0])
    test = row[1].decode("utf-8")
    str_list = [str(s) for s in json.loads(test)]
    maps[row[0]] = str_list
    for i in str_list:
        fileMaps[i] = row[2].decode("utf-8")
 # 打印map内容

print(fileMaps)
print(maps)
#shuffle
random.shuffle(chunkIds)
print(chunkIds)

def group_chunk_ids(lst, chunk_size):
    """
    将列表按照指定大小分组
    """
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

shuffle_ids = group_chunk_ids(chunkIds, 4)
print(shuffle_ids)

shuffle_files = []
for group_ids in shuffle_ids:
    group_files = []
    for chunk_id in group_ids:
        group_files.extend(maps[chunk_id])

    random.shuffle(group_files)
    shuffle_files.extend(group_files)


print(shuffle_files)
tar = None
for i in shuffle_files:
    tar_name = fileMaps[i]
    if tar is None or tar_name != tar.name :
        tar = tarfile.open(mount + "/pack/" + tar_name, "r:")
    print(tar.extractfile(i).read())
# str = [x.decode() for x in shuffle_files]
# print(str)

