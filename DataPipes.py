from time import sleep

import torchdata.datapipes.iter
from torchdata.datapipes.iter import FileLister, FileOpener, StreamReader
from torchdata.datapipes.iter import FileLister, FileOpener
import tarfile

datapipe1 = FileLister("/mnt/jfs2/pack", "imagenet_4M_*")
print(list(datapipe1))
print("*********tgz file************\n")
datapipe2 = FileOpener(datapipe1, mode="b")
# print(list(datapipe2))
tar_decompress_dp = torchdata.datapipes.iter.Decompressor(datapipe2, file_type="tar")
for _, stream in tar_decompress_dp:
    print(stream.read())
tar_loader_dp = datapipe2.load_from_tar().shuffle()
# print(list(tar_loader_dp))
list = list(tar_loader_dp)


sleep(5)
for name, stream in list[:3]:
    # sleep(2)
    print(name)
    print(stream.read())

print("end")
sleep(5)
for name, stream in list[3:]:
    # sleep(2)
    print(name)
    print(stream.read())

# list2 = list.shuffle()
# sleep(5)
#
# for name, stream in list2:
#
#     print(name)
#     # print(stream.read())

# tar = tarfile.open("/data/beeond/data/test.tgz")
# print(tar.getmember("/data/beeond/data/test.tgz"))
# dp = StreamReader(tar_loader_dp)
# print(list(dp))