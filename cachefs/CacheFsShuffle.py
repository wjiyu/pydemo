import json
import logging
import os.path
import random
import tarfile
import threading
import math
import zlib
from cachefs.CacheFsDatabase import CacheFsDatabase
from PIL import Image, TarIO
from skimage.color import gray2rgb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ChunkStruct:
    def __init__(self, name, chunk_id):
        self.name = name
        self.chunk_id = chunk_id


class ChunkInfo:
    def __init__(self):
        #chunkid-files map
        self.chunk_maps = {}
        # self.files = []
        #chunk name-files map
        self.name_maps = {}
        self.chunk_ids = []
        self.lock = threading.Lock()


class CacheFsShuffle:
    def __init__(self, path, conf, group_size=4, work=50):
        """
        初始化函数，对类字段进行初始化
        :param path: 数据路径
        :param conf: 数据库配置信息
        :param group_size: 组大小（随机打乱时每组文件数量）
        """
        self.path = path
        self.conf = conf
        self.shuffle_files = []
        self.file_maps = {}
        self.group_size = group_size
        self.work = work if work < 100 else 100
        self.cachefs_database = CacheFsDatabase(self.conf)

    def group_chunk_ids(self, lst):
        """
        对文件chunk id进行分组，每组self.group_size个文件
        :param lst: 文件chunk id列表
        :return: 分组后的chunk id列表
        """
        return [lst[i:i + self.group_size] for i in range(0, len(lst), self.group_size)]

    def query_chunks(self):
        sql = 'SELECT name, chunkid FROM jfs_chunk_file WHERE name LIKE %s'
        result = self.cachefs_database.fetch(os.path.basename(self.path), sql)
        chunk_array = []
        for row in result:
            chunk_array.append(ChunkStruct(row[0].decode("utf-8"), row[1]))

        return chunk_array

    def split_array(self, chunks):
        if chunks is None:
            return None
        if self.work <= 0:
            return None

        size = math.ceil(len(chunks) / self.work)
        slices = [chunks[i:i + size] for i in range(0, len(chunks), size)]
        return slices

    def query_files(self):
        chunks = self.query_chunks()
        slices = self.split_array(chunks)
        if slices is None:
            return
        info = ChunkInfo()
        threads = []
        for i in range(len(slices)):
            t = threading.Thread(target=self.parsing_files, args=(slices[i], info))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        return info

    def parsing_files(self, chunks, info):
        chunk_ids = []
        slice_maps = {}
        for chunk in chunks:
            if len(chunk.name) == 0:
                logging.error(f"Corrupt entry with empty name: inode {chunk.name}")
                continue
            logging.debug(f"name: {chunk.name}")
            index = chunk.name.rfind("_")
            if index < 0:
                logging.debug(f"filter non dataset info: {chunk.name}")
                continue
            sub_name = chunk.name[:index]
            if os.path.basename(self.path) == sub_name:
                chunk_ids.append(chunk.chunk_id)
                slice_maps[chunk.chunk_id] = chunk.name
            else:
                continue

        result = self.cachefs_database.query_slices(chunk_ids)

        # file_list = []
        chunk_maps = {}
        name_maps = {}
        for row in result:
            decompress_files = zlib.decompress(row[1]).decode("utf-8")  # row[1].decode("utf-8")
            files = [str(s) for s in json.loads(decompress_files)]
            # file_list.extend(files)
            chunk_maps[row[0]] = files
            name = slice_maps.get(row[0])
            name_maps.update({file:name for file in files})

        with info.lock:
            # info.files.extend(file_list)
            info.chunk_maps.update(chunk_maps)
            info.name_maps.update(name_maps)
            info.chunk_ids.extend(chunk_ids)

        # info.lock.acquire()
        # try:
        #     info.files.extend(file_list)
        #     info.chunk_maps.update(chunk_maps)
        # finally:
        #     info.mux.release()

    def shuffle(self):
        """
        对文件进行随机打乱，并返回打乱后的文件列表和文件映射信息
        :return: 打乱后的文件列表和文件映射信息
        """

        # sql
        # sql = 'SELECT b.chunkid, b.files, a.name FROM jfs_chunk_file a INNER JOIN jfs_slice_file b ON a.chunkid = b.chunkid WHERE a.name LIKE %s'
        #
        # result = self.cachefs_database.fetch(os.path.basename(self.path), sql)
        info = self.query_files()
        self.file_maps = info.name_maps
        # chunk_ids = []
        # maps = {}
        # for row in result:
        #     chunk_ids.append(row[0])
        #     decompress_files = zlib.decompress(row[1]).decode("utf-8")  # row[1].decode("utf-8")
        #     files = [str(s) for s in json.loads(decompress_files)]
        #     maps[row[0]] = files
        #     for i in files:
        #         self.file_maps[i] = row[2].decode("utf-8")
        random.shuffle(info.chunk_ids)
        shuffle_ids = self.group_chunk_ids(info.chunk_ids)
        for group_ids in shuffle_ids:
            group_files = []
            for chunk_id in group_ids:
                group_files.extend(info.chunk_maps.get(chunk_id))
            random.shuffle(group_files)
            self.shuffle_files.extend(group_files)
        return self.shuffle_files, self.file_maps

    @staticmethod
    def extract_file(tar_path, file_name):
        """
        从压缩文件中提取指定文件
        :param tar_path: 压缩文件路径
        :param file_name: 文件名
        :return: 文件内容
        """
        with tarfile.open(tar_path, "r:") as tar:
            return tar.extractfile(file_name).read()

    @staticmethod
    def extract_image(path, file_name):
        """
        从一个.tar文件中提取图像并返回图像对象
        :param path: .tar文件路径
        :param fileName: 目标图像文件名
        :return: Image对象
        """
        with TarIO.TarIO(path, file_name) as fp:
            im = Image.open(fp).convert("RGB")
            if im.mode == 'L':
                im = gray2rgb(im)
        return im

# def extractTarFile(self, tarPath):
#     myDatabase = MyDatabase(self.conf)
#     mount = myDatabase.queryMount()
#     tar = None
#     for i in self.shuffle_files:
#         tar_name = self.fileMaps[i]
#         if tar is None or tar_name != tar.name:
#             tar = tarfile.open(mount + "/pack/" + tar_name, "r:")
#         print(tar.extractfile(i).read())
#         return tar.extractfile(i).read()
