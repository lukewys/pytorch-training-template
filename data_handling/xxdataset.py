import glob
from torch.utils.data import Dataset

class XXDataset(Dataset):

    def __init__(self, data_dir, single_sample=False):
        self.data_list = sorted(glob.glob(data_dir + '/*.npy'))
        self.single_sample = single_sample

    def __len__(self):
        if self.single_sample:
            return 1
        else:
            return len(self.data_list)

    def __getitem__(self, item):
        if self.single_sample:  # single_sample refers to debugging mode where only one data sample is used
            item = 0
        # xxx
        return src_single, tgt_single
