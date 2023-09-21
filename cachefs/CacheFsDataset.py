import concurrent.futures
import glob
import multiprocessing
import os
import tarfile
import threading
import random

import numpy as np
import torchvision
from PIL import Image, TarIO
from matplotlib import pyplot as plt
from skimage.color import gray2rgb
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder

from CacheFsShuffle import CacheFsShuffle
from CacheFsDatabase import CacheFsDatabase
import torchvision.datasets as datasets

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tarfile
import os
import io
import cv2

from torchvision.datasets.folder import default_loader

from torchvision.transforms.functional import to_tensor, to_pil_image
from torchdata.datapipes.iter import FileOpener
import time

from cachefs.CacheFsContainerIO import CacheFsContainerIO
from cachefs.CacheFsChunkIO import CacheFsChunkIO
from cachefs.CacheFsTarfile import CacheFsTarfile


# import accimage


class CacheFsDataset(Dataset):
    """
    cachefs 自定义dataset
    :param root_dir: 数据路径
    :param conf: 数据库配置信息
    :param transform: 转换
    :param group_size: 组大小（可以参数调优，chunk聚合小文件数量越多，最优group_size:2, chunk聚合小文件数量少，最优group_size:4）
    """
    def __init__(self, root_dir, conf, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.cachefs = CacheFsShuffle(root_dir, conf)
        self.shuffle_files, self.file_maps = self.cachefs.shuffle()
        self.default_load = self.load_files if self.cachefs.group_size == 2 else self.read_files
        database = CacheFsDatabase(conf)
        self.mount = database.query_mount()
        self.lock = threading.Lock()
        self.chunk_stream = {}

    def __len__(self):
        return len(self.shuffle_files)

    def __getitem__(self, idx):

        chunk_path = os.path.sep.join([self.mount, "pack", os.path.basename(self.root_dir), self.file_maps.get(self.shuffle_files[idx])])

        if not os.path.isfile(chunk_path):
            print(chunk_path + ' mount path does not exist!')
            return None

        try:
            # with tarfile.open(chunk_path, 'r') as tar:
            #     file = tar.extractfile(self.shuffle_files[idx])
            #     image = Image.open(file).convert("RGB")
            #     if self.transform is not None:
            #         image = self.transform(image)
            #     return image

            with TarIO.TarIO(chunk_path, self.shuffle_files[idx]) as fp, Image.open(fp).convert("RGB") as image:
                if self.transform is not None:
                    image = self.transform(image)
                return image

        except Exception as e:
            print("exception: ", str(e), chunk_path, self.shuffle_files[idx])
            pass

    def chunk_stream_wrapper(self, chunk_paths):
        datapipe = FileOpener(list(chunk_paths), mode="rb").load_from_tar()
        with self.lock:
            self.chunk_stream = {file[0]: file[1] for file in datapipe}

    def extract_files(self, tar, files, index_maps, data):
        for file in files:
            with tar.extractfile(file) as fp, Image.open(fp).convert("RGB") as image:
                if self.transform is not None:
                    image = self.transform(image)
                data[index_maps.get(file)] = image
        tar.close()

    def extract_stream(self, key, files, index_maps, data):
        datapipe = FileOpener([key], mode="rb").load_from_tar().filter(
            filter_fn=lambda file_name: file_name[0] in files)
        for dp in datapipe:
            with Image.open(dp[1]).convert("RGB") as image:
                if self.transform is not None:
                    image = self.transform(image)
                data[index_maps.get(dp[0])] = image

    def tar_io_files(self, chunk_path, file, index, datas):
        with TarIO.TarIO(chunk_path, file) as fp, Image.open(fp).convert("RGB") as image:
            if self.transform is not None:
                image = self.transform(image)
            datas[index] = image

    def load_image(self, tar, fp, file, index, datas, lock):
        try:
            with lock:
                with fp, Image.open(fp, "r").convert("RGB") as image:
                    if self.transform is not None:
                        image = self.transform(image)
                    try:
                        datas[index] = image
                    except Exception as e:
                        print("data except: ", str(e))
        except Exception as e:
            print("read exception: ", e)
        finally:
            tar.removemember(file)
            if not tar.member_maps:
                tar.close()

    def load_images(self, tar, file_objects, index_maps, datas, files):
        try:
            for file, obj in file_objects.items():
                with obj as fp, Image.open(fp, "r").convert("RGB") as image:
                    if self.transform is not None:
                        image = self.transform(image)
                    datas[index_maps.get(file)] = image
            tar.removemember(files)
        except Exception as e:
            print("load_images exception: ", str(e))
            raise e
        finally:
            if not tar.member_maps:
                tar.close()


    ############################高并发最优方案2##################################
    def load_files(self, chunk_path, files, index_maps, datas):
        with CacheFsChunkIO(chunk_path, files) as members:
            for file, fp in members.members.items():
                with Image.open(fp).convert("RGB") as image:
                    if self.transform is not None:
                        image = self.transform(image)
                    datas[index_maps.get(file)] = image

    ######################高并发最优方案1###############################
    def read_files(self, chunk_path, files, index_maps, datas):
        try:
            if chunk_path not in self.chunk_stream:
                with CacheFsChunkIO(chunk_path) as chunk_io:
                    with self.lock:
                        if chunk_path not in self.chunk_stream:
                            self.chunk_stream[chunk_path] = chunk_io
                    for file, fp in chunk_io.getmembers(files).items():
                        with Image.open(fp).convert("RGB") as image:
                            if self.transform is not None:
                                image = self.transform(image)
                        datas[index_maps.get(file)] = image
            else:
                with open(chunk_path, "rb") as fh:
                    for file, fp in self.chunk_stream.get(chunk_path).getmembers(files, fh).items():
                        with Image.open(fp).convert("RGB") as image:
                            if self.transform is not None:
                                image = self.transform(image)
                        datas[index_maps.get(file)] = image
        except Exception as e:
            print("read files exception: ", str(e))
            raise e
        finally:
            try:
                self.chunk_stream.get(chunk_path).deletemembers(files)
                if not self.chunk_stream.get(chunk_path).members:
                    self.chunk_stream.pop(chunk_path, None)
            except Exception as e:
                print("delete chunk stream error: ", str(e))
                raise e

    def load_infos(self, key, file, index_map, datas):
        try:
            if key not in self.chunk_stream:
                with CacheFsChunkIO(key) as tar:
                    with self.lock:
                        self.chunk_stream[key] = tar

            objs = self.chunk_stream.get(key)

            if not objs:
                with TarIO.TarIO(key, file) as obj, Image.open(obj).convert("RGB") as image:
                    if self.transform is not None:
                        image = self.transform(image)
                    datas[index_map.get(file)] = image
            else:
                with open(key, "rb") as fh:
                    obj = objs.members.get(file)
                    obj.fh = fh
                    with Image.open(obj).convert("RGB") as image:
                        if self.transform is not None:
                            image = self.transform(image)
                        datas[index_map.get(file)] = image
        except Exception as e:
            print("load infos exception: ", str(e))
            raise e
        finally:
            pass

    def __getitems__(self, batch_idx):
        chunk_file_maps = {}
        index_maps = {}
        for index, idx in enumerate(batch_idx):

            file = self.shuffle_files[idx]
            chunk_path = os.path.sep.join([self.mount, "pack", os.path.basename(self.root_dir), self.file_maps.get(file)])

            if not os.path.isfile(chunk_path):
                print(chunk_path + ' mount path does not exist!')
                raise Exception(chunk_path + ' mount path does not exist!')

            chunk_file_maps.get(chunk_path).append(file)  if chunk_file_maps.get(chunk_path, [])  else chunk_file_maps.setdefault(chunk_path, [file])
            index_maps[file] = index

        try:
            datas = [0] * len(batch_idx)
            ###################多进程方案############################
            # data_list = multiprocessing.Manager().list()
            # process_pool = multiprocessing.Pool(processes=len(chunk_file_maps.keys()))
            # for key, files in chunk_file_maps.items():
            #     process_pool.apply_async(self.read_files, args=(key, files, index_maps), callback=data_list.extend)
            #
            # process_pool.close()
            # process_pool.join()
            # datas = [data for _, data in data_list]
            # thread_futures = []
            # with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunk_file_maps.keys())) as executor:
            #     for key, files in chunk_file_maps.items():
            #         thread_future = executor.submit(self.read_files, key, files, index_maps, datas)
            #         thread_futures.append(thread_future)
            # concurrent.futures.wait(thread_futures)
            ###############高并发最优方案####################
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunk_file_maps.keys())) as executor:
                thread_futures = []
                for key, files in chunk_file_maps.items():
                ##############################高并发方案5#####################################################
                # fh = open(key, "rb")
                # fhs.append(fh)
                # local = threading.local()
                #     for file in files:
                #         thread_future = executor.submit(self.load_infos, key, file, index_maps, datas)
                #         thread_futures.append(thread_future)
                # concurrent.futures.wait(thread_futures)
            ###############################高并发方案4#################################################
            # for file in files:
            # with obj as fp, Image.open(fp).convert("RGB") as image:
            #     if self.transform is not None:
            #         image = self.transform(image)
            #     try:
            #         datas[index_maps.get(file)] = image
            #     except Exception as e:
            #         print("data except: ", str(e))

            # thread_future = executor.submit(self.tar_io_files, key, file, index_maps.get(file), datas)
            # thread_futures.append(thread_future)
            #########################高并发方案3-首选###############################################
                    # self.read_files(key, files, index_maps, datas)
                    #######################最优方案1#####################################
                    thread_future = executor.submit(self.default_load, key, files, index_maps, datas)
                    #######################最优方案2#####################################
                    # thread_future = executor.submit(self.load_files, key, files, index_maps, datas)
                    thread_futures.append(thread_future)
                concurrent.futures.wait(thread_futures)

            #########################高并发方案2###############################################

            # tar = self.chunk_stream.get(key)
            # file_objects = tar.extractfiles(files)
            # lock = threading.Lock()
            # for file, obj in file_objects.items():
            #     # with obj as fp, Image.open(fp).convert("RGB") as image:
            #     #     if self.transform is not None:
            #     #         image = self.transform(image)
            #     #     try:
            #     #         datas[index_maps.get(file)] = image
            #     #     except Exception as e:
            #     #         print("data except: ", str(e))
            #
            #     thread_future = executor.submit(self.load_image, tar, obj, file, index_maps.get(file), datas, lock)
            #     thread_futures.append(thread_future)

            ###################高并发方案1##############################################
            # tar = self.chunk_stream.get(key)
            # file_objects = tar.extractfiles(files)
            # thread_future = executor.submit(self.load_images, tar, file_objects, index_maps, datas, files)
            # thread_futures.append(thread_future)

            # concurrent.futures.wait(thread_futures)
            return datas

        except Exception as e:
            print("exception: ", str(e))
            pass


    def thread_process(self, batch_idx):
        try:
            data = [None] * len(batch_idx)
            thread_futures = []
            for i, idx in enumerate(batch_idx):
                thread_future = self.thread_pool.submit(self.fetch_datas, idx, i, data)
                # thread_future.add_done_callback(lambda f, index=i: update_result(index, f.result()))
                thread_futures.append(thread_future)

            concurrent.futures.wait(thread_futures)
            print(len(data))
            return data
        except Exception as e:
            print(e)
            raise e


    # def thread_data(self, file, index, data):
    #     try:
    #         start_time = time.time()
    #
    #         # chunk_stream = self.chunk_stream[idx].filter(filter_fn=lambda file_name: file in file_name[0])
    #         # stream = next(iter(chunk_stream))
    #         # print(stream)
    #         # print("time: ", time.time() - start_time)
    #         with Image.open(stream[1]) as image:
    #             image = image.convert("RGB")
    #             if self.transform is not None:
    #                 image = self.transform(image)
    #             print(index)
    #             data[index] = image
    #     except Exception as e:
    #         print("thread: ", e)
    #     finally:
    #         pass

    def fetch_datas(self, batch_idx):
        try:
            data = [0] * len(batch_idx)
            thread_futures = []
            for index, idx in enumerate(batch_idx):
                file = self.mount + "/pack/" + os.path.basename(self.root_dir) + os.path.sep \
                       + self.file_maps.get(self.shuffle_files[idx]) + os.path.sep + self.shuffle_files[idx]

                # stream_key = self.mount + "/pack/" + os.path.basename(self.root_dir) + os.path.sep + self.file_maps.get(
                #     self.shuffle_files[idx])

                # start_time = time.time()
                # while True:
                #     if time.time() - start_time < 5:
                #         if stream_key not in self.chunk_stream:
                #             pass
                #         else:
                #             break
                #     else:
                #         print("time out: ", stream_key)
                #         self.chunk_stream[stream_key] = FileOpener([stream_key], mode="rb").load_from_tar()
                #         break

                thread_future = self.thread_pool.submit(self.thread_data, file, index, data)
                thread_futures.append(thread_future)
            concurrent.futures.wait(thread_futures)
            return data
        except Exception as e:
            print(e)
            raise e
