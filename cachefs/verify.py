import concurrent
import io
import itertools
import multiprocessing
import os.path
import tarfile
from typing import Iterable, Optional

import torchvision.io
from PIL.ContainerIO import ContainerIO
from torch.utils.data.datapipes.utils.common import StreamWrapper
from torchdata.dataloader2 import DataLoader2
from torchdata.dataloader2.adapter import Shuffle
from torchvision.utils import make_grid
from torch.utils.data.datapipes.utils.common import get_file_binaries_from_pathnames
from torchdata.datapipes.iter import FileLister, FileOpener, StreamReader

from cachefs.CacheFsChunkFile import CacheFsChunkFile
from cachefs.CacheFsChunkIO import CacheFsChunkIO
from cachefs.CacheFsTarfile import CacheFsTarfile


class Example:
    def __init__(self, num_workers):
        self._num_workers = num_workers

    def iterate_workers(self):
        workers = itertools.cycle(range(self._num_workers))
        for worker in workers:
            print(f"Processing worker {worker}")
            # Perform some task using the worker


# example = Example(3)
# # example.iterate_workers()
# test = itertools.cycle(range(10))
# print(test)
# for i in itertools.cycle(range(10)):
#     print(i)
from torch.utils.data import Dataset, DataLoader


class BatchProcess(Dataset):
    def __init__(self):
        self.files = ["files", "file1"]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.files[idx]

        return [self.files[i] for i in idx]


# batchProcess = BatchProcess()
# print(batchProcess[0])
# print(batchProcess[0, 1])
#
# set1 = {4,5}
# my_set = {1,2,3}
#
# my_set |= set1
#
#
# if my_set:
#     for item in my_set:
#         print(item)
# else:
#     print("The set is empty")
#
# files = {}
# files[my_set] = ["test"]
# files[set1] = ["test1"]
#
# print(files.get(4))
# my_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
#
# # Remove a specific key from the dictionary
# key_to_remove = 'key2'
# my_dict.pop("key3")
# del my_dict[key_to_remove]
#
# # Print the updated dictionary
# print(my_dict)
from torchdata.datapipes.iter import FileLister, FileOpener, IterableWrapper, Mapper
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import dill
from PIL import Image, TarIO

# Iterable TarArchiveLoader读取数据
# datapip1 = FileLister("/mnt/jfs/pack/imagenet_4M", "*")
# datapipe2 = FileOpener(datapip1, mode="rb")
# data = datapipe2.load_from_tar()
# print(data)
# plt.figure()
# for file in datapipe2.load_from_tar():
#     # Open the image file
#     # Create a PIL Image object from the file data
#     image = Image.open(file[1])
#     # Display the image
#     # image.show()
#
#     # show_images_batch(image)
#     # grid = make_grid(image)
#     plt.imshow(image)
#     plt.axis('off')
#     plt.ioff()
#     plt.show()
#     break
# plt.show()

# dp = IterableWrapper(range(10)).shuffle()
# map_dp_1 = Mapper(dp, lambda x: x + 1)
# map_dp_2 = dp.map(lambda x: x + 1)
#
# print(list(map_dp_1))
# print(list(map_dp_2))
# filter_dp = map_dp_1.filter(lambda x: x % 2 == 0)
# print(list(filter_dp))
#
# d1 = DataLoader2(dp, [Shuffle(False)])
# for i in d1:
#     print(i)
import torchdata.datapipes as dp

# datapipe = dp.iter.FileLister(["/mnt/jfs/pack/imagenet_4M"])  # .filter(filter_fn=lambda filename: filename.endswitch(""))
# datapipe = dp.iter.FileOpener(datapipe, mode="rb")
# datapipe = datapipe.load_from_tar()
# print(datapipe)
# tar = tarfile.open("/mnt/jfs/pack/imagenet_4M/imagenet_4M_0")
# plt.figure()
# with tar.extractfile("imagenet_4M/imagenet_3/imagenet_2/ILSVRC2012_test_00000027.JPEG") as f:
#     image = Image.open(f)
#     image = image.convert("RGB")
#     plt.imshow(image)
#     plt.axis('off')
#     plt.ioff()
#     plt.show()

# for data in datapipe.load_from_tar():
#     print(data[0])
#     # print(data[1].read())
#     with tar.extractfile("imagenet_4M/imagenet_3/imagenet_2/ILSVRC2012_test_00000027.JPEG") as f:
#         image = Image.open(f)
#         image = image.convert("RGB")
#         plt.imshow(image)
#         plt.axis('off')
#         plt.ioff()
#         plt.show()
# # image.show()
# # plt.imshow(image)
# # plt.axis('off')
# # plt.ioff()
# # plt.show()
#     break

# from torchdata.datapipes.iter import FileLister, FileOpener
# from PIL import Image
#
# # Create a FileLister datapipe to list all files in the tar package
# file_lister = FileLister("/mnt/jfs/pack/imagenet_4M", "*")
#
# # Create a FileOpener datapipe to open the image files
# file_opener = FileOpener(file_lister)
#
# # Create an IterableWrapper datapipe to iterate over the image files
# image_files = IterableWrapper(file_lister, file_opener)
#
# # Iterate over each image file
# for image_file in image_files:
#     # Open the image file
#     with image_file.open() as file:
#         # Create a PIL Image object from the file data
#         image = Image.open(file)
#         # Display the image
#         image.show()


# with TarIO.TarIO("/mnt/jfs/pack/imagenet_4M/imagenet_4M_0", "imagenet_4M/imagenet_3/imagenet_2/ILSVRC2012_test_00000027.JPEG") as fp, Image.open(fp) as image:
#     image = image.convert("RGB")
#     print(image)
#     print(image.show())

from torchdata.datapipes.iter import FSSpecFileLister
import time
# datapipe = FSSpecFileLister(root="/mnt/jfs/pack/imagenet_4M")
# file_dp = datapipe.open_files_by_fsspec()
# print(file_dp)
# for file in file_dp:
#     print(file)
#     break
import torch
# dp = IterableWrapper(["/mnt/jfs/pack/imagenet_4M"])
# dp = dp.open_files_by_fsspec(mode="rb", anon=True).load_from_tar()
from torchvision.datasets.folder import default_loader

# datapipe = dp.iter.FileLister(["/mnt/jfs/pack/imagenet_4M"])  # .filter(filter_fn=lambda filename: filename.endswitch(""))
path = "/mnt/jfs/pack/imagenet_4M/imagenet_4M_1"
path1 = "/mnt/jfs/pack/imagenet_4M/imagenet_4M_0"
path3 = "imagenet_4M/imagenet_3/imagenet_2/ILSVRC2012_test_00000030.JPEG"
path4 = "imagenet_4M/imagenet_1/imagenet_2/ILSVRC2012_test_00000029.JPEG"


path5 = "imagenet_4M/imagenet_2/imagenet_2/ILSVRC2012_test_00000035.JPEG"

map = {"file1": [path1], "file2": path}


print(map)
print("test: ", os.path.sep.join(["/mnt", "pack", "image"]))
map.get("file3", []).append("add") if map.get("file3", []) else map.setdefault("file3", ["sub"])
map.get("file1", []).append("add") if map.get("file1", []) else map.setdefault("file1", ["sub"])
print(map)

with CacheFsTarfile.open(path1, "r") as tar:
    files = tar.extractfiles([path3, path4])
    print(files)

import tarfile
import io
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# def open_image(fp):
#     with Image.open(fp) as image:
#         print("test")
#         datas.append(image)
#         print(datas)
#     fp.close()
#

# def getTarIO(tarfile, cont):
#     fh = open(tarfile, "rb")
#     return ContainerIO(fh, cont.offset, cont.length)

start = time.time()
list = [path3, path4]
tar1 = CacheFsChunkIO(path1, list)
print(tar1)
print("tar: ", tar1.members)
# global_members = tar1.members
map = {path1:tar1.members}
print(map)
# print(tar1)
image = Image.open(tar1.members.get(path3))
print(image)
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ima = transform(image)
print(ima)
# # datas = []
# # # thread_futures = []
# plt.figure()
# for file, fp in tar1.members.items():
#     image = Image.open(fp)
#     plt.imshow(image)
#     plt.axis('off')
#     plt.ioff()
#     plt.show()
# print("time1: ", time.time() - start)
tar1.close()
obj = map.get(path1).get(path3)
fp = open(path1, mode="rb")
test = CacheFsChunkFile(fp, obj.offset, obj.length)
print(test)
image = Image.open(test, formats=["JPEG"])
# plt.imshow(image)
# plt.axis('off')
# plt.ioff()
# plt.show()

from PIL import Image

# Create a new image with a specified size and color
width = 500
height = 300
color = (255, 0, 0)  # RGB color value for red
new_image = Image.new('RGB', (width, height), color)

# Display the image
new_image.show()
plt.imshow(new_image)
plt.axis('off')
plt.ioff()
plt.show()


import cv2
import numpy as np

def open_image(file):
    # Read the file object using numpy's frombuffer() function
    buffer = np.frombuffer(test.fh.read(), np.uint8)
    # Decode the image using cv2.imdecode()
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image

# Example usage with file object
with open(path1, 'rb') as file:
    image = open_image(file)
    plt.imshow(image)
    plt.axis('off')
    plt.ioff()
    plt.show()
    # Display or process the image as needed


from PIL import Image

# 读取图片数据
# with open("path/to/image.jpg", "rb") as fh:
#     image_data = fh.read()

# 使用frombytes方法创建图像对象
image = Image.frombytes("RGB", (width, height), test.fh.read())
plt.imshow(image)
plt.axis('off')
plt.ioff()
plt.show()
# 执行图像操作
# ...

# 关闭图像对象
image.close()









import cv2

def cv2_open(filename):
    image = cv2.imread(filename)
    return image

def cv2_resize(image, size):
    resized_image = cv2.resize(image, size)
    return resized_image

def cv2_rotate(image, angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image

def cv2_save(image, filename):
    cv2.imwrite(filename, image)

# Example usage
filename = 'image.jpg'
image = cv2_open(filename)
resized_image = cv2_resize(image, (800, 600))
rotated_image = cv2_rotate(resized_image, 45)
cv2_save(rotated_image, 'output.jpg')


import multiprocessing
import tarfile

# def read_tar_file(tar_file):
#     images = []
#     with tarfile.open(tar_file, 'r') as tar:
#         for member in tar.getmembers():
#             if member.isfile() and member.name.endswith('.jpg'):
#                 image_data = tar.extractfile(member).read()
#                 images.append(image_data)
#     return images
#
# if __name__ == '__main__':
#     tar_files = [path, path1]  # List of tar file paths
#
#     # Create a global list to store the image data
#     global_list = multiprocessing.Manager().list()
#
#     # Create a process pool
#     pool = multiprocessing.Pool()
#
#     # Iterate over the tar files
#     for tar_file in tar_files:
#         # Apply_async schedules the function to be executed in a process
#         # The result is appended to the global list
#         pool.apply_async(read_tar_file, args=(tar_file,), callback=global_list.extend)
#
#     # Close the pool to prevent any more tasks from being submitted
#     pool.close()
#
#     # Wait for all processes to complete
#     pool.join()
#
#     # Convert the global list to a regular list
#     image_data_list = list(global_list)

    # Return the image data list
    # return image_data_list

import concurrent.futures
import tarfile

def read_tar_file(tar_file, index):
    images = []
    with tarfile.open(tar_file, 'r') as tar:
        for member in tar.getmembers():
            if member.isfile():
                image_data = tar.extractfile(member)
                image = Image.open(image_data).convert("RGB")
                images.append((index, image))
    return images

if __name__ == '__main__':
    tar_files = [path, path1]  # List of tar file paths
    max_workers = 4  # Maximum number of concurrent processes

    # Create a global list to store the image data
    global_list = []

    # Create a thread pool executor with a limited number of workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit the tasks to the executor
        futures = [executor.submit(read_tar_file, tar_file, index) for index, tar_file in enumerate(tar_files)]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

        # Get the results from the completed tasks
        for future in futures:
            images = future.result()
            global_list.extend(images)

    # Sort the global list based on the index
    print(global_list)
    # sorted_list = sorted(global_list, key=lambda x: x[0])
    #
    # # Extract the image data from the sorted list
    # image_data_list = [data for _, data in sorted_list]
    #
    # # Return the image data list
    # print(image_data_list)



# import multiprocessing
# import tarfile
#
# def read_tar_file(tar_file, index):
#     images = []
#     with tarfile.open(tar_file, 'r') as tar:
#         for member in tar.getmembers():
#             if member.isfile():
#                 image_data = tar.extractfile(member)
#                 image = Image.open(image_data).convert("RGB")
#                 images.append((index, image))
#     return images
#
# if __name__ == '__main__':
#     tar_files = [path, path1]  # List of tar file paths
#
#     # Create a global list to store the image data
#     global_list = multiprocessing.Manager().list()
#
#     # Create a process pool
#     pool = multiprocessing.Pool(processes=1)
#
#     # Iterate over the tar files
#     for index, tar_file in enumerate(tar_files):
#         # Apply_async schedules the function to be executed in a process
#         # The result is appended to the global list
#         pool.apply_async(read_tar_file, args=(tar_file, index), callback=global_list.extend)
#
#     # Close the pool to prevent any more tasks from being submitted
#     pool.close()
#
#     # Wait for all processes to complete
#     pool.join()
#
#     print(global_list)
#
#     # Sort the global list based on the index
#     sorted_list = sorted(global_list, key=lambda x: x[0])
#
#     # Extract the image data from the sorted list
#     image_data_list = [data for _, data in sorted_list]

    # Return the image data list
    # return image_data_list
    # print(image_data_list)








import multiprocessing

def producer(conn):
    """该函数将在生产者进程中执行"""
    for i in range(10):
        conn.send({"key": "value"})
    conn.close()

def consumer(conn):
    """该函数将在消费者进程中执行"""
    while True:
        item = conn.recv()
        if item is None:
            break
        print(item)

if __name__ == '__main__':
    # 创建管道
    conn1, conn2 = multiprocessing.Pipe()
    # 创建生产者进程
    p1 = multiprocessing.Process(target=producer, args=(conn1,))
    # 创建消费者进程
    p2 = multiprocessing.Process(target=consumer, args=(conn2,))
    # 启动进程
    p1.start()
    p2.start()
    # 等待进程结束
    p1.join()
    # 发送结束信号
    conn1.send(None)
    p2.join()
















# with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#     for file, fp in tar1.members.items():
#         extract_thread = executor.submit(open_image, fp)
#         thread_futures.append(extract_thread)
# Create a thread pool for reading files
# with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#     # Submit tasks to the thread pool
#     while not file_queue.empty():
#         file, extracted_file = file_queue.get()
#         print(file)
#         thread_futures.append(executor.submit(read_file, file, extracted_file))

# concurrent.futures.wait(thread_futures)
# plt.figure()
# for file, fp in tar1.members.items():
#     datas.append(Image.open(fp))
#     # plt.imshow(data)
#     # plt.axis('off')
#     # plt.ioff()
#     # plt.show()
# print("time1: ", time.time() - start)

# start = time.time()
# tar1 = TarIO.TarIO(path, path5)
# print(tar1)
# image = Image.open(tar1)
# plt.imshow(image)
# plt.axis('off')
# plt.ioff()
# plt.show()
# print(image)
# print("time1: ", time.time() - start)

# Open the tar file
# start = time.time()
# tar = tarfile.open(path1, 'r')
#
# # Extract the image file from the tar file
# image_file = tar.extractfile(path3)
# image = Image.open(image_file)
# print("time: ", time.time() - start)


# Read the image data
# image_data = image_file.read()
#
# image_data = torchvision.io.read_image(image_data)
# print(image_data)

# image = torch.load(io.BytesIO(image_data))
# plt.imshow(image)
# plt.axis('off')
# plt.ioff()
# plt.show()

# tensor = torch.ByteTensor(list(image_data))
# width = 224
# height = 224
# channels = 224
# tensor = tensor.view(channels, height, width)
# tensor = tensor.float() / 255.0
# transform = transforms.ToPILImage()
# image = transform(tensor)
# # Convert the image data to a PyTorch tensor
# # image_np = np.frombuffer(image_data, dtype=np.uint8)
# # image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
#
# # Display the image
# plt.imshow(image)
# plt.axis('off')
# plt.ioff()
# plt.show()
#
# # Close the tar file
# tar.close()





# To directly convert the data read by  `tar.extractfile`  in Python to image data that can be directly used by PyTorch, you can follow these steps:
#
# 1. Import the necessary libraries:
# import tarfile
# import torch
# import torchvision.transforms as transforms
# import io
# from PIL import Image
# 2. Extract the file data using  `tar.extractfile` :
# tar = tarfile.open('your_tar_file.tar', 'r')
# file_data = tar.extractfile('path_to_file_inside_tar')
# 3. Read the file data and convert it to a PyTorch tensor:
# # Read the file data
# data = file_data.read()
#
# # Convert the data to a PyTorch tensor
# tensor = torch.ByteTensor(list(data))
# 4. Reshape the tensor to match the image dimensions:
# # Specify the image dimensions
# width = 224
# height = 224
# channels = 3
#
# # Reshape the tensor to match the image dimensions
# tensor = tensor.view(channels, height, width)
# 5. Normalize and transform the tensor to be compatible with PyTorch image format:
# # Normalize the tensor values to be in the range [0, 1]
# tensor = tensor.float() / 255.0
#
# # Create a transform to convert the tensor to a PyTorch image
# transform = transforms.ToPILImage()
#
# # Apply the transform to the tensor
# image = transform(tensor)
# Now, you have the image data in a PyTorch image format that can be directly used by PyTorch for further processing or analysis.

















#
# import tarfile
# from PIL import Image
# from io import BytesIO
# import asyncio
# import aiofiles
# import aiohttp
#
# # 创建线程局部存储
# tls = asyncio.local()
#
# # 全局字典用于存储已打开的tar文件
# tar_files = {}
#
# async def open_tar(tar_path):
#     if tar_path not in tar_files:
#         async with aiofiles.open(tar_path, 'rb') as f:
#             tar_files[tar_path] = tarfile.open(fileobj=f)
#
# async def extract_file(tar_path, file_path):
#     # 打开tar文件，如果当前线程没有打开过该tar文件
#     if not hasattr(tls, 'tar') or tls.tar.name != tar_path:
#         await open_tar(tar_path)
#         tls.tar = tar_files[tar_path]
#
#     # 抽取文件数据
#     file_data = tls.tar.extractfile(file_path)
#
#     # 处理文件数据
#     await process_image(file_data)
#
# async def process_image(file_data):
#     # 从文件数据中读取图像
#     image = Image.open(BytesIO(file_data.read()))
#
#     # 处理图像数据
#     # ...
#
#     # 关闭图像文件
#     image.close()
#
# async def main():
#     # 指定tar文件和文件路径列表
#     tar_files = [path, path1]
#     file_paths = [path3, path4]
#
#     # 创建连接池
#     async with aiohttp.ClientSession() as session:
#         # 创建任务列表
#         tasks = []
#         for tar_file, file_path in zip(tar_files, file_paths):
#             task = asyncio.create_task(extract_file(tar_file, file_path))
#             tasks.append(task)
#
#         # 等待任务完成
#         await asyncio.gather(*tasks)
#
#     # 关闭tar文件
#     for tar in tar_files.values():
#         tar.close()
#
# if __name__ == '__main__':
#     asyncio.run(main())



# import concurrent.futures
# import queue
# import tarfile
#
# # Open the tar file
# tar = tarfile.open('/mnt/jfs/pack/imagenet_4M/imagenet_4M_1', 'r')
#
# # Create a queue to hold the extracted files
# file_queue = queue.Queue()
# thread_futures = []
# # Function to extract files and put them into the queue
# def extract_files():
#     for file in tar.getnames():
#         extracted_file = tar.extractfile(file)
#         file_queue.put((file, extracted_file))
#
# # Function to read the extracted files
# def read_file(file, extracted_file):
#     try:
#         # Process the file content here
#         content = extracted_file.read()
#         print(f"Content of {file}: {content}")
#
#     except Exception as e:
#         print(f"Error reading file: {file} - {e}")
#
# # Start extracting files in a separate thread
# extract_thread = concurrent.futures.ThreadPoolExecutor(max_workers=2).submit(extract_files)
# thread_futures.append(extract_thread)
# # Create a thread pool for reading files
# with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#     # Submit tasks to the thread pool
#     while not file_queue.empty():
#         file, extracted_file = file_queue.get()
#         print(file)
#         thread_futures.append(executor.submit(read_file, file, extracted_file))
#
# concurrent.futures.wait(thread_futures)
# print(file_queue)
# # Close the tar file
# tar.close()











# tar = tarfile.open(path1, 'r')
# print(tar.getmembers())
#
# fp = tar.extractfile("imagenet_4M/imagenet_3/imagenet_2/ILSVRC2012_test_00000030.JPEG")
#
# image =  Image.open(fp).convert("RGB")
#
# fp1 = tar.extractfile("imagenet_4M/imagenet_3/imagenet_2/ILSVRC2012_test_00000029.JPEG")
#
# image1 =  Image.open(fp1).convert("RGB")
#
# stream = StreamWrapper(tar.extractfile("imagenet_4M/imagenet_3/imagenet_2/ILSVRC2012_test_00000030.JPEG"),
#                        name="imagenet_4M/imagenet_3/imagenet_2/ILSVRC2012_test_00000030.JPEG")
# # print(stream.read())
# image = Image.open(stream).convert("RGB")
#
# plt.figure()
# # for key, value in test2:
# #     # cont = file[1].read()
# #     image = Image.open(value)
# # image = image.convert("RGB")
# #     # image = default_loader(file[1])
# #     image = torch.load(io.BytesIO(file[1].read()))
# plt.imshow(image)
# plt.axis('off')
# plt.ioff()
# plt.show()
#
# # with tarfile.open(path1, 'r') as tar:
# #     file = tar.extractfile(path3)
# #     image = Image.open(file).convert("RGB")
# #     print(image)
#
#
# start = time.time()
# datapipe = FileOpener([path, path1], mode="rb")
# for i in datapipe:
#     print(i)
datapipe = datapipe.load_from_tar().fork(num_instances=2)
# print("fork: ", datapipe)
# maps = {path1: datapipe}
# datapipe = datapipe.filter(filter_fn=lambda filename: path3 in filename[0])
# print(datapipe)
# print("data: ", next(iter(maps.get(path1).filter(filter_fn=lambda filename: path3 in filename[0]))))
# for i in datapipe:
#     print(i)
#
# datapipe1 = datapipe.filter(filter_fn=lambda filename: path4 in filename[0])
# for i in datapipe:
#     print(i)
# # test1 = {file[0]:file[1] for file in datapipe}
# print(time.time() - start)
#
#
# def use_pipe(maps):
#     print("process data: ", next(iter(maps.get(path1).filter(filter_fn=lambda filename: path3 in filename[0]))))
#
#
# process = multiprocessing.Process(target=use_pipe, args=(maps,))
# process1 = multiprocessing.Process(target=use_pipe, args=(maps,))
#
# process.start()
# process1.start()
# print("procedata: ", next(iter(maps.get(path1).filter(filter_fn=lambda filename: path3 in filename[0]))))
# print("proce data: ", next(iter(maps.get(path1).filter(filter_fn=lambda filename: path3 in filename[0]))))
# process1.join()
# process.join()

# dp = IterableWrapper([path3]) \
#         .open_files_by_iopath()
# print(dp)
# # datapipe = datapipe.load_from_tar()
# dp = StreamReader(datapipe)
# print(list(dp))
# for i in dp:
#     print(i)
#     break
# # d1 = DataLoader(datapipe)
# # stream = StreamWrapper(open("/mnt/jfs/pack/imagenet_4M/imagenet_4M_0", "rb"))
# print(stream)
#
# stream_maps = {file[0] :file[1] for file in datapipe}
# print(stream_maps)
import fsspec

# start = time.time()
# dp1 = IterableWrapper([path1])
# dp2 = dp1.open_files(mode="rb").load_from_tar()
# test2={file[0]:file[1] for file in dp2}
# print(time.time() - start)

# start = time.time()
# dp = IterableWrapper([path1])
# dp3 = dp.open_files_by_fsspec(mode="rb", anon=True).load_from_tar(mode="r|")
# test3={file[0]:file[1] for file in dp3}
# print(time.time() - start)

# plt.figure()
# for key, value in test2:
#     # cont = file[1].read()
#     image = Image.open(value)
#     image = image.convert("RGB")
#     # image = default_loader(file[1])
#     # image = torch.load(io.BytesIO(file[1].read()))
#     plt.imshow(image)
#     plt.axis('off')
#     plt.ioff()
#     plt.show()
#     break


# for i in dp2:
#     print(i)
#
# dp1 = dp1.list_files_by_fsspec()
# for i in dp1:
#     print(i)
# print(dp1)
#
# dp1 = dp1.open_files_by_fsspec(mode="rb", anon=True).load_from_tar(mode="r|")
# print(dp1)
# for i in dp1:
#     print(i)
# plt.figure()

# for file in datapipe:
#     print(file[0])
#     print(stream_maps.get(file[0]))
#     # cont = file[1].read()
#     image = Image.open(stream_maps.get(file[0]))
#     image = image.convert("RGB")
#     # image = default_loader(file[1])
#     # image = torch.load(io.BytesIO(file[1].read()))
#     plt.imshow(image)
#     plt.axis('off')
#     plt.ioff()
#     plt.show()
#     break
import time
# path1 = "imagenet_4M/imagenet_3/imagenet_2/ILSVRC2012_test_00000027.JPEG"
# cont = file[1].read()
# stream = stream_maps.get(path1)

# def get_file_binaries_from_pathnames1(pathnames: Iterable, mode: str, encoding: Optional[str] = None):
#     if not isinstance(pathnames, Iterable):
#         pathnames = [pathnames, ]
#
#     if mode in ('b', 't'):
#         mode = 'r' + mode
#     print(pathnames)
#     for pathname in pathnames:
#         print("pathname: ", pathname)
#         if not isinstance(pathname, str):
#             raise TypeError("Expected string type for pathname, but got {}"
#                             .format(type(pathname)))
#         yield pathname, StreamWrapper(open(pathname, mode, encoding=encoding))
#
# path2: Iterable = "/mnt/jfs/pack/imagenet_4M/imagenet_4M_0"
# stream = get_file_binaries_from_pathnames1(path2, "rb")
# print(stream)
#
# for i in stream:
#     print(i)

# stream = StreamWrapper(open(path2, "rb", encoding=None))

# with Image.open(stream) as image:
#     image = image.convert("RGB")
# # image = default_loader(file[1])
# # image = torch.load(io.BytesIO(file[1].read()))
# plt.imshow(image)
# plt.axis('off')
# plt.ioff()
# plt.show()
#
# stream.close()

# time.sleep(3)
# path2 = "/mnt/jfs/pack/imagenet_4M/imagenet_4M_0/imagenet_4M/imagenet_3/imagenet_2/ILSVRC2012_test_00000030.JPEG"
# StreamWrapper(open(path1, "rb"))
# stream = stream_maps.get(path1)
# image = Image.open(stream)
# image = image.convert("RGB")
# # image = default_loader(file[1])
# # image = torch.load(io.BytesIO(file[1].read()))
# plt.imshow(image)
# plt.axis('off')
# plt.ioff()
# plt.show()

#
# set1 = {1, 2, 3}
# set2 = {4, 5, 6}
# set1.update(set2)
# print(set1)
