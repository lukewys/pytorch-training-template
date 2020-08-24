from torch.utils.data.dataloader import DataLoader
from functools import partial

from .xxdataset import XXDataset
from .collate_fn import data_collate_fn


def get_data_loader(batch_size, XXX, shuffle=True, num_workers=0, drop_last=False, single_sample=False):
    dataset = XXDataset(XXXX, single_sample=single_sample)

    collate_fn = partial(
        collate_fn,
        XXXX)

    return DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers,
        drop_last=drop_last, collate_fn=collate_fn, pin_memory=True)
