import glob
import os
import tarfile

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from CacheFsDataset import MyDataset


def show_images_batch(image):
    grid = make_grid(image)
    plt.imshow(grid.numpy().transpose(1, 2, 0))


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MyDataset(root_dir='/mnt/cachefs/pack/imagenet_4M', conf='/home/wjy/db.conf', transform=transform)

    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, num_workers=4)

    plt.figure()
    for image in enumerate(dataloader):
        print(image)
        show_images_batch(image)
        plt.axis('off')
        plt.ioff()
        plt.show()
    plt.show()



    # # specify the directory containing tar files
    # dir_path = "/data/beeond/data/imagenet/imagenet_100G" #"/data/beeond/data/imagenet/imagenet_100G"
    # # create an empty dictionary to store the mapping
    # tar_map = {}
    # files = []
    # # iterate over all tar files in the directory
    # for tar_file in glob.glob(os.path.join(dir_path, "imagenet_100G_16008")):
    #     # open the tar file
    #     with tarfile.open(tar_file, "r") as tar:
    #         # iterate over all files in the tar file
    #         for member in tar.getmembers():
    #             # add the file name and tar file name to the dictionary
    #             tar_map[member.name] = os.path.basename(tar_file)
    #             files.append(member.name)
    # # print the dictionary
    # print(tar_map)
    # print(files)