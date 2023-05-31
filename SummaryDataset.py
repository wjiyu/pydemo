from torch.utils.data import IterableDataset, DataLoader
import math

class SummaryDataset(IterableDataset):

    def __init__(self,
                 file_path: str,
                 rank,
                 world_size
                 ):
        super(SummaryDataset).__init__()
        self.file_path = file_path
        self.info = self._get_file_info(file_path)
        self.start = self.info['start']
        self.end = self.info['end']

        self.rank = rank
        self.world_size = world_size

        self.per_worker = int(math.floor((self.end - self.start) / float(self.world_size)))
        self.iter_start = self.start + self.rank * self.per_worker
        self.iter_end = min(self.iter_start + self.per_worker, self.end)

    def __iter__(self):
        sample_iterator = self._sample_generator(self.iter_start, self.iter_end)
        return sample_iterator

    def __len__(self):
        return self.iter_end - self.iter_start

    def _get_file_info(self,
                       file_path
                       ):
        info = {
            "start": 1,
            "end": 0,
            "id_colum": 0,
            "article_colum": 1,
            "summary_colum": 2
        }
        with open(file_path, 'r') as fin:
            for _ in enumerate(fin):
                info['end'] += 1
        return info

    def _sample_generator(self, start, end):
        id_c, art_c, sum_c = self.info['id_colum'], self.info['article_colum'], self.info['summary_colum']
        with open(self.file_path, 'r') as fin:
            for i, line in enumerate(fin):
                if i < start: continue
                if i >= end: return StopIteration()
                items = line.strip().split('\t')
                sample = {"id": items[id_c], "article": items[art_c], "summary": items[sum_c]}
                yield sample


def train_worker(rank, args):
    # 子进程

    train_dataset = SummaryDataset("/data/beeond/data/wjy/raw_stories", rank, 5)
    rain_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=64, num_workers=0)


train_worker(100, 0)