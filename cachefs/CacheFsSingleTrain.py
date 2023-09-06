import shutil

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
import time

from cachefs.Decompress import Decompress
from cachefs.FolderDataset import FolderDataset


def show_images_batch(image):
    grid = make_grid(image)
    plt.imshow(grid.numpy().transpose(1, 2, 0))


if __name__ == '__main__':

    start = time.time()

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        dataset = Decompress(root_dir='/mnt/jfs/pack/imagenet_100W_4K', conf='/home/wjy/db.conf', transform=transform)

        dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=16)

        # plt.figure()
        for idx, image in enumerate(dataloader):
            pass
        #     print(idx)
        #     print(image)
        #     # show_images_batch(image)
        #     grid = make_grid(image)
        #     plt.imshow(grid.numpy().transpose(1, 2, 0))
        #     plt.axis('off')
        #     plt.ioff()
        #     plt.show()
        # plt.show()

        print("Execution time:", time.time() - start, "seconds")
    finally:
        # dataset.thread_pool.shutdown()
        shutil.rmtree("/var/jfsCache/imagenet_100W_4K")
