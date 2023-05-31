import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from MyDataset import MyDataset


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

    dataset = MyDataset(root_dir='/mnt/jfs2/pack/imagenet_4M', conf='/home/wjy/db.conf', transform=transform)

    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, num_workers=2)

    plt.figure()
    for image in dataloader:
        print(image)
        show_images_batch(image)
        plt.axis('off')
        plt.ioff()
        plt.show()
    plt.show()
