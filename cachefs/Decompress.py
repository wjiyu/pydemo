import concurrent.futures
import shutil
import signal
import threading
from PIL import Image, TarIO
from torch.utils.data import Dataset
from CacheFsShuffle import CacheFsShuffle
from CacheFsDatabase import CacheFsDatabase
import os
from torchvision.datasets.folder import default_loader
import time


class Decompress(Dataset):
    def __init__(self, root_dir, conf, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.cachefs = CacheFsShuffle(root_dir, conf, 4)
        self.shuffle_files, self.file_maps = self.cachefs.shuffle()
        self.size = len(self.shuffle_files)
        database = CacheFsDatabase(conf)
        self.mount = database.query_mount()
        self.out_dir = "/var/jfsCache"
        self.extracted_chunk = set()
        self.lock = threading.Lock()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=20)
        # chunk_path to thread map
        self.thread_maps = {}
        # signal.signal(signal.SIGINT, self.signal_handler)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        chunk_path = self.mount + "/pack/" + os.path.basename(self.root_dir) + os.path.sep + self.file_maps.get(
            self.shuffle_files[idx])

        out_path = self.out_dir + os.path.sep + self.shuffle_files[idx]

        if not os.path.isfile(chunk_path):
            print(chunk_path + ' mount path does not exist!')
            return None

        for index in range(1, 5):
            try:
                image = self.loader_image(chunk_path, out_path)
                return image
            except Exception as e:
                if index < 5:
                    print(f"retry {index} loader: ", str(e), out_path)
                    time.sleep(index * index / 1000)
                    continue
                else:
                    raise e

    def unpack_archive(self, chunk_path):
        try:
            shutil.unpack_archive(chunk_path, self.out_dir, "tar")
        except FileExistsError as e:
            self.extracted_chunk.remove(chunk_path)

    def loader_image(self, chunk_path, out_path):
        try:
            if not os.path.isfile(out_path):
                if not chunk_path in self.extracted_chunk:
                    with self.lock:
                        self.extracted_chunk.add(chunk_path)
                        self.thread_maps[chunk_path] = self.thread_pool.submit(self.unpack_archive, chunk_path)
                future = self.thread_maps.get(chunk_path)
                future.result()
            image = default_loader(out_path)
            if self.transform is not None:
                image = self.transform(image)
            return image
        except Exception as e:
            raise e


    def loader(self, chunk_path, out_path):
        try:
            if not os.path.isfile(out_path):
                if not chunk_path in self.extracted_chunk:
                    with self.lock:
                        self.extracted_chunk.add(chunk_path)
                self.unpack_archive(chunk_path)

            image = default_loader(out_path)
            if self.transform is not None:
                image = self.transform(image)
            return image
        except Exception as e:
            raise e

