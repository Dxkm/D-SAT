import os
import cv2
import numpy as np
from glob import glob
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


def dataloader(config, random_state=42):
    """     80%  for Train     10%  for Val     10%  for Test     """

    # Data loading code
    img_ids = glob(os.path.join(config['root_dir'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    test_size = int(round((10 / 100) * len(img_ids)))
    train_val_img_ids, test_img_ids = train_test_split(img_ids, test_size=test_size, random_state=random_state)
    train_img_ids, val_img_ids = train_test_split(train_val_img_ids, test_size=test_size, random_state=random_state)

    train_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.RandomRotate90(p=0.5),
        A.GaussianBlur(),
        A.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(),
    ])

    val_test_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize()
    ])

    train_dataset = DataProcessing(img_ids=train_img_ids,
                                   img_dir=os.path.join(config['root_dir'], 'images'),
                                   mask_dir=os.path.join(config['root_dir'], 'masks'),
                                   img_ext=config['img_ext'],
                                   mask_ext=config['mask_ext'],
                                   num_classes=config['num_classes'],
                                   transform=train_transform,
                                   )

    val_dataset = DataProcessing(img_ids=val_img_ids,
                                 img_dir=os.path.join(config['root_dir'], 'images'),
                                 mask_dir=os.path.join(config['root_dir'], 'masks'),
                                 img_ext=config['img_ext'],
                                 mask_ext=config['mask_ext'],
                                 num_classes=config['num_classes'],
                                 transform=val_test_transform,
                                 )

    test_dataset = DataProcessing(img_ids=test_img_ids,
                                  img_dir=os.path.join(config['root_dir'], 'images'),
                                  mask_dir=os.path.join(config['root_dir'], 'masks'),
                                  img_ext=config['img_ext'],
                                  mask_ext=config['mask_ext'],
                                  num_classes=config['num_classes'],
                                  transform=val_test_transform,
                                  )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=config['num_workers'], drop_last=False)
    return train_loader, val_loader, test_loader


class DataProcessing(Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_ext = self.img_ext
        mask_ext = self.mask_ext

        img = cv2.imread(os.path.join(self.img_dir, img_id + img_ext))

        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i), img_id + mask_ext),
                                   cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}
