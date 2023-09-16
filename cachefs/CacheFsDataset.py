import glob
import os
import tarfile

from PIL import Image, TarIO
from skimage.color import gray2rgb
from torch.utils.data import Dataset
from CacheFsShuffle import CacheFsShuffle
from CacheFsDatabase import CacheFsDatabase
import torchvision.datasets as datasets

from cachefs.CacheFsTarfile import CacheFsTarfile


class CacheFsDataset(Dataset):
    def __init__(self, root_dir, conf, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.cachefs = CacheFsShuffle(root_dir, conf, 4)
        self.shuffle_files, self.file_maps = self.cachefs.shuffle()
        self.size = len(self.shuffle_files)
        database = CacheFsDatabase(conf)
        self.mount = database.query_mount()


    def get_tar_fils(self):
        # specify the directory containing tar files
        dir_path = self.root_dir  # "/data/beeond/data/imagenet/imagenet_100G"
        # create an empty dictionary to store the mapping
        tar_map = {}
        files = []
        # iterate over all tar files in the directory
        for tar_file in glob.glob(os.path.join(dir_path, "*")):
            # open the tar file
            with tarfile.open(tar_file, "r") as tar:
                # iterate over all files in the tar file
                for member in tar.getmembers():
                    # add the file name and tar file name to the dictionary
                    tar_map[member.name] = os.path.basename(tar_file)
                    files.append(member.name)
        # # print the dictionary
        # print(tar_map)
        return files, tar_map

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        chunk_path = self.mount + "/pack/" + os.path.basename(self.root_dir) + os.path.sep \
                     + self.file_maps.get(self.shuffle_files[idx])

        # chunk_path = "/data/beeond/data/imagenet/" + os.path.basename(self.root_dir) + os.path.sep \
        #              + self.file_maps.get(self.files[idx])

        if not os.path.isfile(chunk_path):
            print(chunk_path + ' mount path does not exist!')
            return None

        try:

            # with TarIO.TarIO(chunk_path, self.shuffle_files[idx]) as fp, Image.open(fp).convert("RGB") as image:
            #     if self.transform is not None:
            #         image = self.transform(image)
            #     return image

            with CacheFsTarfile.open(chunk_path, 'r') as tar:
                # for file in tar.getmembers():
                #     print(file)
                #     print(tar.extractfile(file).read())
                # print(tar.getmember(self.shuffle_files[idx]))
                im = tar.extractfile(self.shuffle_files[idx])
                image =  Image.open(im)
                print(image)
                if self.transform:
                    image = self.transform(image)
                return image
        except Exception as e:
            print("exception: ", str(e), chunk_path, self.shuffle_files[idx])
            pass

        # image = self.cachefs.extract_image(chunk_path, self.shuffle_files[idx])





# class TarImageNetDataset(datasets.DatasetFolder):
#     def __init__(self, root, transform=None, target_transform=None,
#                  loader=default_loader, is_valid_file=None):
#         super(TarImageNetDataset, self).__init__(root, loader, None,
#                                                   transform=transform,
#                                                   target_transform=target_transform,
#                                                   is_valid_file=is_valid_file)
#         self.tar = tarfile.open(root, 'r')
#
#     def __getitem__(self, index):
#         tarinfo = self.samples[index]
#         fileobj = self.tar.extractfile(tarinfo)
#         img = self.loader(fileobj)
#         if self.transform is not None:
#             img = self.transform(img)
#         target = self.target_transform(tarinfo[1])
#         return img, target
#
#     def __len__(self):
#         return len(self.samples)