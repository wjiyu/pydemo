import glob
import os
import tarfile

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
