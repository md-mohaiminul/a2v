import torch
import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __len__(self):
        files = glob.glob(os.path.join(self.root_dir, '*.pkl'))
        return len(files)

    def __getitem__(self, index):
        a_file = open(self.root_dir+'/dict_' + str(index) + '.pkl', 'rb')
        dict = pickle.load(a_file)
        return dict
# dataset = CustomDataset(root_dir='./dataset')
# print(dataset.__len__())
# dict = dataset.__getitem__(1)
# print(dict["video"].shape, dict["audio"].shape)
#
# data_loader = DataLoader(dataset = dataset, batch_size = 16, shuffle = True)
# dict =next(iter(data_loader))
# print(dict["video"].shape, dict["audio"].shape)
#
# for batch_idx, dict in enumerate(data_loader):
#     print(batch_idx, dict["audio"].shape, dict["video"].shape)
