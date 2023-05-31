import matplotlib.pyplot as plt
import torch
from scipy.odr import Model
from torch import nn
from torch.distributed import optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from MyDataset import MyDataset

num_epochs = 5  # 5轮
batch_size = 50  # 50步长
learning_rate = 0.01  # 学习率0.01


def show_images_batch(image):
    grid = make_grid(image)
    plt.imshow(grid.numpy().transpose(1, 2, 0))


def get_dataloader(rank, world_size, root_dir, conf, transform=None, batch_size=32):
    """获取分布式数据集"""
    dataset = MyDataset(root_dir, conf, transform=transform)

    # 分布式采样器
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    # 数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=sampler)

    return dataloader


def get_device(rank):
    """获取分布式设备"""
    return torch.device("cuda:%d" % rank)


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


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # dataset = MyDataset(root_dir='/mnt/jfs2/pack/imagenet_4M', conf='/home/wjy/db.conf', transform=transform)
    #
    # dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, num_workers=2)

    # 分布式初始化
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # 获取数据加载器和设备
    dataloader = get_dataloader(rank, world_size, '/mnt/jfs2/pack/imagenet_4M', '/home/wjy/db.conf',
                                transform=transform, batch_size=4)
    device = get_device(rank)

    # 构建模型
    # Initialize the model
    # model = create_model()
    model = Model()

    # model = resnet18()
    model.to(device)

    # 利用PyTorch的分布式数据并行功能，实现分布式训练
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    # 指定优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 开始训练
    for epoch in range(num_epochs):
        for input_data in dataloader:
            input_data = input_data.to(device)
            output = model(input_data)
            # 计算损失和梯度并更新模型参数
            loss = criterion(output, input_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # plt.figure()
    # for image in dataloader:
    #     print(image)
    #     show_images_batch(image)
    #     plt.axis('off')
    #     plt.ioff()
    #     plt.show()
    # plt.show()
