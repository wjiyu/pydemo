import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import make_grid
import time


def show_images_batch(image):
    grid = make_grid(image)
    plt.imshow(grid.numpy().transpose(1, 2, 0))


if __name__ == '__main__':

    start = time.time()

    # transform = transforms.Compose([
    #     transforms.Resize(256),  # 调整图像大小为256x256像素
    #     transforms.CenterCrop(224),  # 从中心裁剪出224x224像素的图像
    #     transforms.ToTensor(),  # 将图像转换为张量
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
    # ])

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder('/data/beeond/data/imagenet_100W_4K', transform=transform)

    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=16)

    # plt.figure()
    for idx, (image, target) in enumerate(dataloader):
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
