import torch  # top-level pytorch package and tensor library
import torchvision
import matplotlib.pyplot as plt
import numpy as np

print(torch.__version__)

print(torch.cuda.is_available())
print(torch.version.cuda)

batch_size_train = 128  # 设置训练集的 batch size，即每批次将参与运算的样本数
batch_size_test = 128  # 设置测试集 batch size

train_set = torchvision.datasets.MNIST('./dataset_mnist', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,)
                                           )
                                       ])
)


test_set = torchvision.datasets.MNIST('./dataset_mnist', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,)
                                           )
                                       ])
)


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=True)

print(len(train_set))  # train_set 中的样本总数
print(train_set.train_labels)  # train_set中的样本标签
print(train_set.train_labels.bincount())  # 查看每一个标签有多少样本

print(train_set.classes)  # 查看 train_set 的样本类别
print(len(train_set.classes))  # 查看train_set中有所少种类别
print(train_set.class_to_idx)  # 查看样本类别和样本标签的对应关系

sample = next(iter(train_set))  # get an item from train_set

print("For each item in train_set: \n\n \t type: ", type(sample))  # tuple (image, label)
print("\t Length: ", len(sample), '\n')  # 2

image, label = sample # unpack the sample

print("For each image: \n\n \t type: ", type(image)) # rank-3 tensor
print("\t shape: ", image.shape, '\n') # [channel, height, width] = [1, 28, 28]  Note: 仅有3维！
print("For each label: \n\n \t type: ", type(label), '\n')



print("Let's check an image: \n ")
plt.imshow(image.squeeze(), cmap='gray')
print(f'label: {label}')


train_loader_plot = torch.utils.data.DataLoader(
    train_set, batch_size=40
)  # 假设一个批次有40个样本

batch = next(iter(train_loader_plot))
print("type(batch): \t", type(batch))  # list [images, labels]
print("len(batch): \t", len(batch), "\n") # 2

images, labels = batch

print("type(images): \t", type(images))  # rank-4 tensor
print("images.shape: \t", images.shape)  # [batch_size, channel, height, width] = [10, 1, 28, 28]

print("type(labels): \t", type(labels))  # rank-1 tensor
print("labels.shape: \t", labels.shape)  # size=batch size


# 画出第一个批次的样本

grid = torchvision.utils.make_grid(images, nrow=10)  # make a grid of images (grid is a tensor)
plt.figure(figsize=(12,12))
plt.imshow(np.transpose(grid, (1,2,0))) # np.transpose permutes the dimensions
print(f'labels: {labels}')
