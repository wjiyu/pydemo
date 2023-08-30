import glob
import os
import tarfile

from torch.utils.data import Dataset
from CacheFsShuffle import CacheFsShuffle
from CacheFsDatabase import CacheFsDatabase


class MyDataset(Dataset):
    def __init__(self, root_dir, conf, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.myShuffles = CacheFsShuffle(root_dir, conf, 4)
        shuffle_files, file_maps = self.get_tar_fils() #self.myShuffles.shuffle()
        self.files = shuffle_files
        self.size = len(self.files)
        self.file_maps = file_maps

        my_database = CacheFsDatabase(conf)
        self.mount = my_database.query_mount()

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
        # chunk_path = self.mount + "/pack/" + os.path.basename(self.root_dir) + os.path.sep \
        #              + self.file_maps[self.files[idx]]

        chunk_path = "/data/beeond/data/imagenet/" + os.path.basename(self.root_dir) + os.path.sep \
                     + self.file_maps[self.files[idx]]
        if not os.path.isfile(chunk_path):
            print(chunk_path + ' mount path does not exist!')
            return None

        image = self.myShuffles.extract_image(chunk_path, self.files[idx])
        if self.transform:
            image = self.transform(image)

        return image


