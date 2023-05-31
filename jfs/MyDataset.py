import os
from torch.utils.data import Dataset
from MyShuffle import MyShuffles
from MyDatabase import MyDatabase


class MyDataset(Dataset):
    def __init__(self, root_dir, conf, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.myShuffles = MyShuffles(root_dir, conf, 4)
        shuffle_files, file_maps = self.myShuffles.shuffle()
        self.files = shuffle_files
        self.size = len(self.files)
        self.file_maps = file_maps

        my_database = MyDatabase(conf)
        self.mount = my_database.query_mount()

        print(self.files)
        print(self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        chunk_path = self.mount + "/pack/" + self.file_maps[self.files[idx]]
        if not os.path.isfile(chunk_path):
            print(chunk_path + ' does not exist!')
            return None

        image = self.myShuffles.extract_image(chunk_path, self.files[idx])
        if self.transform:
            image = self.transform(image)

        return image


