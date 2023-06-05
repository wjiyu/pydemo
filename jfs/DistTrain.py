import argparse
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.distributed import optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models import resnet18
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from MyDataset import MyDataset
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_epochs = 5  # 5轮
batch_size = 32  # 50步长
learning_rate = 0.01  # 学习率0.01


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def show_images_batch(image):
    grid = make_grid(image)
    plt.imshow(grid.numpy().transpose(1, 2, 0))


def get_dataloader(rank, world_size, root_dir, conf, transform=None, batch_size=32):
    """获取分布式数据集"""
    dataset = MyDataset(root_dir, conf, transform=transform)

    # 分布式采样器
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    # 数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1, sampler=sampler)

    return dataloader


def get_device(rank):
    """获取分布式设备"""
    return torch.device("cuda", rank)


def create_model():
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),  # MNIST images are 28x28 pixels
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10, bias=False)  # 10 classes to predict
    )
    return model


def train(rank, dataloader, model, criterion, optimizer, num_epochs=32):

    print("============================  Training  ============================ \n")

    model.train()
    plt.figure()
    for epoch in range(1, num_epochs + 1):
        for index, images in enumerate(dataloader):
            label = [0 for i in range(len(images))]
            image, target = images.to(DEVICE), torch.tensor(label).to(DEVICE)
            output = model(image)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(images)
            show_images_batch(images)
            plt.axis('off')
            plt.ioff()
            plt.show()

            if len(images) != batch_size:
                length = len(dataloader.sampler)
            else:
                length = (index + 1) * len(images)

            print('gpu: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}'.format(rank, epoch, length, len(dataloader.sampler),
                                                                          100. * length / len(dataloader.sampler),
                                                                          loss.item()))
        plt.show()

        print("\n============================  Training Finished  ============================ \n")


def main(rank, world_size):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0, help='node rank for distributed training')
    # parser.add_argument("--word_size", default=1, help="word size")
    args = parser.parse_args()
    # get local_rank from args
    local_rank = args.local_rank

    transform = get_transform()

    # 分布式初始化init_method='env://'  init_method='tcp://10.151.11.61:28765'
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)

    print("local_rank: ", local_rank)

    # 获取数据加载器
    dataloader = get_dataloader(local_rank, world_size, '/mnt/jfs2/pack/imagenet_4M', '/home/wjy/db.conf',
                                transform=transform, batch_size=batch_size)
    # device = get_device(local_rank)

    # 构建模型
    model = resnet18().cuda()

    # 利用PyTorch的分布式数据并行功能，实现分布式训练
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # 指定优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(local_rank, dataloader, model, criterion, optimizer, num_epochs)


if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "10.151.11.54"
    # os.environ["MASTER_PORT"] = "12225"
    world_size = 1
    # main(0, world_size)
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
