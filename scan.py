# # os.listdir
#
# import os
# import subprocess
# import time
# import inspect
#
# dir = '/mnt/node52/imagenet_1000W_4KB/imagenet_62W_4KB_1/imagenet_15W_4KB_1'
#
# # listdir
#
# # start = time.time()
# # result = os.listdir(dir)
# # print(len(result))
# # for file in result:
# #     pass
# # print(time.time() - start)
#
# # shell ls -l -f
#
# # start = time.time()
# # result = subprocess.Popen('ls -1 -f ' + dir, stdout=subprocess.PIPE, shell=True)
# #
# # for file in result.stdout:
# #     pass
# # print(time.time() - start)
#
# # scandir
#
# start = time.time()
# path = os.scandir(dir)
#
# for i in path:
#     pass
# print(time.time() - start)
#
# # os.walk 模块
#
# # start = time.time()
# # for root, dirs, files in os.walk(dir, topdown=False):
# #     pass
# #
# # print(time.time() - start)
#
#
# import os
# import json
# entries = os.scandir('/mnt/node52/imagenet_1000W_4KB/imagenet_62W_4KB_1/imagenet_15W_4KB_1')
# files = []
# directories = []
# count = 0
# for entry in entries:
#    if entry.is_file():
#        ++count
#        #files.append({'name': entry.name})
#    #elif entry.is_dir():
#        #directories.append({'name': entry.name})
# result = {'files': files, 'directories': directories}
# #print(json.dumps(result))
# print(count)


import multiprocessing
import os
import time

from cachefs.CacheFsShuffle import CacheFsShuffle

def hello_from_process():
    print(f'Hello from {os.getpid()}')


# def __iter__(sampler, batch_size) -> Iterator[List[int]]:
#     batch = []
#     for idx in sampler:
#         batch.append(idx)
#         print("test")
#         if len(batch) == batch_size:
#             yield batch
#             batch = []
#             print("count")


if __name__ == '__main__':
    # hellp_process = multiprocessing.Process(target=hello_from_process)
    # hellp_process.start()
    # print(f"hello parent process {os.getpid()}")
    # hellp_process.join()
    #
    # batch = [1, 2, 3, 4]
    #
    #
    # def my_generator():
    #     yield batch
    #     print("hello")
    #     # yield "test"
    #
    #
    # gen = __iter__(batch, 2)  # 创建一个生成器对象
    # print(next(gen))  # 输出: 1
    # print(next(gen))  # 输出: 2
    # # print(next(gen))  # 输出: 3
    start = time.time()
    myShuffles = CacheFsShuffle("/mnt/cachefs/pack/imagenet_100W_4K", '/home/wjy/db.conf', 4)
    files, maps = myShuffles.shuffle()
    print(len(files))
    # print(maps)
    print("Execution time:", time.time() - start, "seconds")


