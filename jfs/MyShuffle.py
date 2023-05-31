import json
import os.path
import random
import tarfile
from MyDatabase import MyDatabase
from PIL import Image, TarIO
from skimage.color import gray2rgb


class MyShuffles:
    def __init__(self, path, conf, group_size=4):
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

    def group_chunk_ids(self, lst):
        """
        对文件chunk id进行分组，每组self.group_size个文件
        :param lst: 文件chunk id列表
        :return: 分组后的chunk id列表
        """
        return [lst[i:i + self.group_size] for i in range(0, len(lst), self.group_size)]

    def shuffle(self):
        """
        对文件进行随机打乱，并返回打乱后的文件列表和文件映射信息
        :return: 打乱后的文件列表和文件映射信息
        """
        my_database = MyDatabase(self.conf)
        result = my_database.fetch(os.path.basename(self.path))
        chunk_ids = []
        maps = {}
        for row in result:
            chunk_ids.append(row[0])
            test = row[1].decode("utf-8")
            str_list = [str(s) for s in json.loads(test)]
            maps[row[0]] = str_list
            for i in str_list:
                self.file_maps[i] = row[2].decode("utf-8")
        random.shuffle(chunk_ids)
        shuffle_ids = self.group_chunk_ids(chunk_ids)
        for group_ids in shuffle_ids:
            group_files = []
            for chunk_id in group_ids:
                group_files.extend(maps[chunk_id])
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
