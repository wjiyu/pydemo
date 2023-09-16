import concurrent.futures
import glob
import os
import tarfile
import threading
import random

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

from torchvision.datasets.folder import default_loader

from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchdata.datapipes.iter import FileOpener
import time

from cachefs.CacheFsTarfile import CacheFsTarfile


# import accimage


class FolderDataset(Dataset):
    def __init__(self, root_dir, conf, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.cachefs = CacheFsShuffle(root_dir, conf, 4)
        self.shuffle_files, self.file_maps = self.cachefs.shuffle()
        self.size = len(self.shuffle_files)
        database = CacheFsDatabase(conf)
        self.mount = database.query_mount()
        self.chunks = set()
        self.lock = threading.Lock()
        self.chunk_stream = {}

        # self.chunk_paths = [self.mount + "/pack/" + os.path.basename(self.root_dir) + os.path.sep + path for path in
        #                     list(set(self.file_maps.values()))]
        # self.chunk_stream = {chunk_path: CacheFsTarfile.open(chunk_path) for chunk_path in self.chunk_paths}
        # self.chunk_stream = FileOpener(self.chunk_paths, mode="rb").load_from_tar().fork(num_instances=self.size,
        #                                                                                  buffer_size=self.size)
        # self.index = 0
        # torchvision.set_image_backend('accimage')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        chunk_path = self.mount + "/pack/" + os.path.basename(self.root_dir) + os.path.sep + self.file_maps.get(
            self.shuffle_files[idx])

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

            # with TarIO.TarIO(chunk_path, self.shuffle_files[idx]) as fp:
            #     image = torch.Tensor(F.to_tensor(F.to_pil_image(accimage.Image(fp))))
            #     print(image)
            #     if self.transform is not None:
            #         image = self.transform(image)
            #     return image
            with TarIO.TarIO(chunk_path, self.shuffle_files[idx]) as fp, Image.open(fp) as image:
                image = image.convert("RGB")
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
        datapipe = FileOpener([key], mode="rb").load_from_tar().filter(filter_fn=lambda file_name: file_name[0] in files)
        for dp in datapipe:
            with Image.open(dp[1]).convert("RGB") as image:
                if self.transform is not None:
                    image = self.transform(image)
                data[index_maps.get(dp[0])] = image


    def tar_files(self, chunk_path, file, index_maps, data):
        with TarIO.TarIO(chunk_path, file) as fp, Image.open(fp).convert("RGB") as image:
            if self.transform is not None:
                image = self.transform(image)
            data[index_maps.get(file)] = image

    def load_image(self, tar, fp, file, index, datas, lock):
        try:
            with fp as fp, Image.open(fp).convert("RGB") as image:
                if self.transform is not None:
                    image = self.transform(image)
                # with lock:
                #     datas[index_maps.get(file)] = image
                return index, image
        finally:
            del tar.member_maps[file]



    def __getitems__(self, batch_idx):
        # chunk_paths = set()
        # print(len(batch_idx))
        chunk_file_maps = dict()
        index_maps = dict()
        for index, idx in enumerate(batch_idx):
            chunk_path = self.mount + "/pack/" + os.path.basename(self.root_dir) + os.path.sep + self.file_maps.get(
                self.shuffle_files[idx])

            # file = chunk_path + os.path.sep + self.shuffle_files[idx]
            file = self.shuffle_files[idx]
            if not os.path.isfile(chunk_path):
                print(chunk_path + ' mount path does not exist!')
                raise Exception(chunk_path + ' mount path does not exist!')

            if chunk_path not in chunk_file_maps:
                # chunk_file_maps[chunk_path] = [self.shuffle_files[idx]]
                chunk_file_maps[chunk_path] = [file]
            else:
                # chunk_file_maps.get(chunk_path).append(self.shuffle_files[idx])
                chunk_file_maps.get(chunk_path).append(file)

            index_maps[file] = index

        try:
            lock = threading.Lock()
            datas = [0] * len(batch_idx)
            with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
                for key, values in chunk_file_maps.items():
                    if key not in self.chunk_stream:
                        with self.lock:
                            self.chunk_stream[key] = CacheFsTarfile.open(key)

                    tar = self.chunk_stream.get(key)
                    file_objects = tar.extractfiles(values)
                    thread_futures = []
                    for file, obj in file_objects.items():
                        # thread_future = executor.submit(self.load_image, tar, obj, file, index_maps.get(file), datas, lock)
                        # thread_futures.append(thread_future)
                        with obj as fp, Image.open(fp).convert("RGB") as image:
                            if self.transform is not None:
                                image = self.transform(image)
                            datas[index_maps.get(file)] = image
                        del tar.member_maps[file]
                    # concurrent.futures.wait(thread_futures)
                    # for thread_future in thread_futures:
                    #     if thread_future.done():
                    #         index, image = thread_future.result()
                    #         datas[index] = image

                    if not tar.member_maps:
                        tar.close()
                    # thread_future = self.thread_pool.submit(self.extract_files, tar, values, index_maps, data)
                    # thread_future.add_done_callback(lambda f, index=i: update_result(index, f.result()))
                    # thread_futures.append(thread_future)

            # if chunk_paths:
            #     stream_maps = {path: FileOpener([path], mode="rb").load_from_tar() for path in chunk_paths}
            #     with self.lock:
            #         self.chunk_stream.update(stream_maps)
            # data = self.fetch_datas(batch_idx)
            # print(datas)
            return datas
            # with tarfile.open(chunk_path, 'r') as tar:
            #     file = tar.extractfile(self.shuffle_files[idx])
            #     image = Image.open(file).convert("RGB")
            #     if self.transform is not None:
            #         image = self.transform(image)
            #     return image
            #
            # all_extracted = all(member.isfile() and member.offset == member.mtime for member in members)

            # with TarIO.TarIO(chunk_path, self.shuffle_files[idx]) as fp:
            #     image = torch.Tensor(F.to_tensor(F.to_pil_image(accimage.Image(fp))))
            #     print(image)
            #     if self.transform is not None:
            #         image = self.transform(image)
            #     return image

            # with TarIO.TarIO(chunk_path, self.shuffle_files[idx]) as fp, Image.open(fp) as image:
            #     image = image.convert("RGB")
            #     if self.transform is not None:
            #         image = self.transform(image)
            #     return image
        except Exception as e:
            # print("exception: ", str(e), chunk_path, self.shuffle_files[idx])
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

    def thread_data(self, file, index, data):
        try:
            start_time = time.time()

            # chunk_stream = self.chunk_stream[idx].filter(filter_fn=lambda file_name: file in file_name[0])
            # stream = next(iter(chunk_stream))
            # print(stream)
            # print("time: ", time.time() - start_time)
            with Image.open(stream[1]) as image:
                image = image.convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
                print(index)
                data[index] = image
        except Exception as e:
            print("thread: ", e)
        finally:
            pass

    def fetch_datas(self, batch_idx):
        try:
            data = [] * len(batch_idx)
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
                # thread_future.add_done_callback(lambda f, index=i: update_result(index, f.result()))
                thread_futures.append(thread_future)

                # stream = next(iter(chunk_stream))
                # try:
                #     with Image.open(stream[1]) as image:
                #         image = image.convert("RGB")
                #         if self.transform is not None:
                #             image = self.transform(image)
                #         data[index]=image
                # finally:
                #     pass
                # chunk_stream.close()
            concurrent.futures.wait(thread_futures)
            return data
        except Exception as e:
            print(e)
            raise e

# image = self.cachefs.extract_image(chunk_path, self.shuffle_files[idx])

# def pil_loader(path: str) -> Image.Image:
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')
#
#
# # TODO: specify the return type
# def accimage_loader(path: str) -> Any:
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)
#
#
# def default_loader(path: str) -> Any:
#     from torchvision import get_image_backend
#     if get_image_backend() == 'accimage':
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)
