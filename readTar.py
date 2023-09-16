import tarfile
import ctypes
from PIL import Image, TarIO
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# lib = ctypes.cdll.LoadLibrary("/home/wjy/juicefs/juicefs-1.0.4/juicefs.so")
# c_char_p_p = ctypes.POINTER(ctypes.c_char_p)
# # example_function = lib.Shuffle
# lib.Shuffle.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
# # example_function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
# lib.Shuffle.restype = c_char_p_p
#
# result= lib.Shuffle("imagenet_4M".encode("utf-8"), "mysql://root:w995219@(10.151.11.61:3306)/juicefs3".encode("utf-8"), 3)
# print(result)
# strings = []
# i = 0
# while True:
#     ptr = result[i]
#     if not ptr:
#         break
#     strings.append(ptr.decode())
#     i += 1
#
# print(strings)
# Open the tar file
tar = tarfile.open("/mnt/jfs2/pack/imagenet_4M_2", "r:")

file_contents = tar.extractfile("imagenet_4M/imagenet_1/ILSVRC2012_test_00000009.JPEG").read()
print(file_contents)
fp = TarIO.TarIO("/mnt/jfs2/pack/imagenet_4M_2", "imagenet_4M/imagenet_1/ILSVRC2012_test_00000009.JPEG")

im = Image.open(fp) #("/mnt/cachefs/imagenet_4M/ILSVRC2012_test_00000009.JPEG")
# print(im)
# plt.imshow(im)
#从Image类转化为numpy array
im = np.asarray(im)
print(im)

plt.imshow(im)
plt.show()
# file_names = tar.getnames()
# print("file name: ", file_names)
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# train_dataset = datasets.ImageFolder(
#     "/mnt/jfs2/pack",
#     transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ]))
#
# print(train_dataset)

# dataset 是数据集对象，indices 是 shuffle 后的索引列表
# subset = Subset(dataset, indices)
#
# # 创建一个 DataLoader 对象用于加载 subset
# dataloader = DataLoader(subset, batch_size=batch_size,
#                         shuffle=False, num_workers=num_workers)

# for file_name in strings:
#     # Extract the file contents
#     # file_contents = tar.extractfile(file_name).read()
#
#     # Print the file name and contents
#     print("File Name: ", file_name)
    # print("File Contents: ", file_contents)
# Extract the file contents
# file_contents = tar.extractfile('imagenet_4M/ILSVRC2012_test_00000001.JPEG').read()

# Print the file contents
# print(file_contents)

# Close the tar file
tar.close()

