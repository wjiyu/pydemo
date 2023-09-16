# from torch.utils.data import DataLoader
#
#
# class CachefsDataLoader(DataLoader):
#     def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None):
#         super(CachefsDataLoader, self).__init__(
#             dataset=dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             collate_fn=collate_fn
#         )
#
#     def __iter__(self):
#         # Implement your custom logic here
#         # You can modify or preprocess the data before yielding the batches
#         for batch in super().__iter__():
#             modified_batch = self.custom_preprocessing(batch)
#             yield modified_batch