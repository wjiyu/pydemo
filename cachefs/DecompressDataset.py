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


class DecompressDataset(Dataset):
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
        if isinstance(idx, int):
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
            else:
                chunk_paths = []
                out_paths = []
                for i in idx:
                    chunk_path = self.mount + "/pack/" + os.path.basename(self.root_dir) + os.path.sep + self.file_maps.get(
                        self.shuffle_files[idx])
                    out_path = self.out_dir + os.path.sep + self.shuffle_files[idx]
                    chunk_paths.append(chunk_path)
                    out_paths.append(out_path)

                for index in range(1, 5):
                    try:
                        images = self.batch_loader_image(chunk_paths, out_paths)
                        return images
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

    def batch_unpack_archive(self, chunk_paths):
        try:
            for chunk_path in chunk_paths:
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

    def batch_loader_image(self, chunk_paths:[], out_paths:[]):
        try:
            undecompress_paths = set()
            future_paths = set()
            for idx, out_path in enumerate(out_paths):
                if not os.path.isfile(out_path):
                    if not chunk_paths[idx] in self.extracted_chunk:
                        undecompress_paths.add(chunk_paths[idx])
                    else:
                        future_paths.add(self.thread_maps.get(chunk_paths[idx]))


            if undecompress_paths:
                with self.lock:
                    self.extracted_chunk |= undecompress_paths
                    future = self.thread_pool.submit(self.batch_unpack_archive, undecompress_paths)
                for path in undecompress_paths:
                    self.thread_maps[path] = future
                future.result()

            if future_paths:
                for future_path in future_paths:
                    future_path.result()

            images = []
            for out_path in out_paths:
                if os.path.isfile(out_path):
                    image = default_loader(out_path)
                    if self.transform is not None:
                        image = self.transform(image)
                    images.append(image)
                else:
                    raise Exception()
            return images
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

