import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

# 判断是否有GPU
from MyData import MyData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 5  # 5轮
batch_size = 50  # 50步长
learning_rate = 0.01  # 学习率0.01

#Cifar 数据集是32*32的图片，如若导入自己数据集记得修改图片宽高数值
img_height = 32
img_width = 32

# 图像预处理
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])




# CIFAR-10 数据集下载
print("dataset start")
time.sleep(10)
# train_dataset = torchvision.datasets.CIFAR10(root='/data/beeond/data/wjy/data/',
#                                              train=True,
#                                              transform=transform,
#                                              download=False)
#
# test_dataset = torchvision.datasets.CIFAR10(root='/data/beeond/data/wjy/data/',
#                                             train=False,
#                                             transform=transforms.ToTensor())

train_dataset = MyData(root='/data/beeond/data/wjy/data/', train=True, transform=transform, download=False)

test_dataset = MyData(root='/data/beeond/data/wjy/data/',
                                            train=False,
                                            transform=transforms.ToTensor())

print("dataset end")

# 数据载入
time.sleep(10)
print("dataloader start")
time.sleep(5)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
print("dataloader end")

# 3x3 卷积定义
def conv3x3(in_channels, out_channels, kernel_size = 3,stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=False)


# Resnet_50  中的残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.mid_channels = out_channels//4
        self.conv1 = conv3x3(in_channels, self.mid_channels, kernel_size=1, stride=stride, padding=0)#Resnet50 中，从第二个残差块开始每个layer的第一个残差块都需要一次downsample
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(self.mid_channels, self.mid_channels)
        self.bn2 = nn.BatchNorm2d(self.mid_channels)
        self.conv3 = conv3x3(self.mid_channels, out_channels,kernel_size=1,padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample_0 = conv3x3(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = self.downsample_0(x)

        out += residual
        out = self.bn3(out)
        out = self.relu(out)
        return out


# ResNet定义
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.conv = conv3x3(3, 64,kernel_size=7,stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(3,2,padding=1)
        self.layer1 = self.make_layer(block, 64, 256, layers[0])
        self.layer2 = self.make_layer(block, 256, 512, layers[1], 2)
        self.layer3 = self.make_layer(block, 512, 1024, layers[2], 2)
        self.layer4 = self.make_layer(block, 1024, 2048, layers[3], 2)
        self.avg_pool = nn.AvgPool2d(3,stride=1,padding=1)
        self.fc = nn.Linear(math.ceil(img_height/32)*math.ceil(img_width/32)*2048, num_classes)

    def make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1):
            downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, kernel_size=3,stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))

        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.max_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view( -1,math.ceil(img_height/32)*math.ceil(img_width/32)*2048)
        return out

#Resnet-50 3-4-6-3 总计(3+4+6+3)*3=48 个conv层 加上开头的两个Conv 一共50层
model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 更新学习率
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 训练数据集
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # 延迟学习率
    if (epoch + 1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# 测试网络模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# S将模型保存
torch.save(model.state_dict(), 'resnet.ckpt')

