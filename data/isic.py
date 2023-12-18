import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def ISIC(config):
    tr_dataset = ISIC2018DatasetFast(mode="tr", one_hot=True, data_dir=config['root_dir'])
    vl_dataset = ISIC2018DatasetFast(mode="vl", one_hot=True, data_dir=config['root_dir'])
    te_dataset = ISIC2018DatasetFast(mode="te", one_hot=True, data_dir=config['root_dir'])

    train_loader = DataLoader(tr_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], drop_last=False, )

    val_loader = DataLoader(vl_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], drop_last=False)

    test_loader = DataLoader(te_dataset, batch_size=config['batch_size'], shuffle=False,
                             num_workers=config['num_workers'], drop_last=False)
    return train_loader, val_loader, test_loader


class ISIC2018DatasetFast(Dataset):
    def __init__(self, mode, one_hot=True, data_dir=None):
        # pre-set variables
        self.data_dir = data_dir if data_dir else '../Data/ISIC2018'
        # input parameters
        self.one_hot = one_hot

        x = np.load(f"{self.data_dir}/X_tr_224x224.npy")
        y = np.load(f"{self.data_dir}/Y_tr_224x224.npy")

        x = torch.tensor(x)
        y = torch.tensor(y)

        if mode == "tr":
            self.image = x[0:1815]
            self.mask = y[0:1815]
        elif mode == "vl":
            self.image = x[1815:1815 + 259]
            self.mask = y[1815:1815 + 259]
        elif mode == "te":
            self.image = x[1815 + 259:2594]
            self.mask = y[1815 + 259:2594]
        else:
            raise ValueError()

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        data_id = idx
        img = self.image[idx]
        msk = self.mask[idx]

        if self.one_hot:
            msk = F.one_hot(torch.squeeze(msk).to(torch.int64))
            msk = torch.moveaxis(msk, -1, 0).to(torch.float)

        sample = {'image': img, 'mask': msk, 'id': data_id}
        return sample
