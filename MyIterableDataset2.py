from torch.utils.data import IterableDataset, DataLoader
import glob


class MyIterableDataset(IterableDataset):

    def __init__(self, file_list):
        super(MyIterableDataset, self).__init__()
        self.file_list = file_list

    def parse_file(self):
        for file in self.file_list:
            print("读取文件：", file)
            with open(file, 'r') as file_obj:
                for line in file_obj:
                    yield line

    def __iter__(self):
        return self.parse_file()


if __name__ == '__main__':
    all_file_list = glob.glob("/data/beeond/data/test01/*.txt")  # 得到datas目录下的所有csv文件的路径
    dataset = MyIterableDataset(all_file_list)

    # 这里batch_size=3，意味着每次读取dataloader都会循环三次dataset
    # drop_last是指到最后，如果凑够了3个数据就返回，如果凑不够就舍弃掉最后的数据
    dataloader = DataLoader(dataset, batch_size=2, drop_last=True)
    for data in dataloader:
        print("")
        print(data)
