import os
import glob
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

# Parameters
INPUT_SIZE = 224

# transform for image
img_transform = transforms.Compose([
    transforms.Resize(
        size=[INPUT_SIZE, INPUT_SIZE],
        interpolation=F.InterpolationMode.BILINEAR
    ),
])
# transform for mask
msk_transform = transforms.Compose([
    transforms.Resize(
        size=[INPUT_SIZE, INPUT_SIZE],
        interpolation=F.InterpolationMode.NEAREST
    ),
])

# preparing input info.
data_prefix = "ISIC_"
target_postfix = "_segmentation"
target_fex = "png"
input_fex = "jpg"
data_dir = 'F:/Data/Data/ISIC2018/'
image_dir = os.path.join(data_dir, "ISIC2018_Task1-2_Training_Input")
mask_dir = os.path.join(data_dir, "ISIC2018_Task1_Training_GroundTruth")

img_dirs = glob.glob(f"{image_dir}/*.{input_fex}")
data_ids = [d.split(data_prefix)[1].split(f".{input_fex}")[0] for d in img_dirs]


def get_img_by_id(Id):
    img_dir = os.path.join(image_dir, f"{data_prefix}{Id}.{input_fex}")
    image = read_image(img_dir, ImageReadMode.RGB)
    return image


def get_msk_by_id(Id):
    msk_dir = os.path.join(mask_dir, f"{data_prefix}{Id}{target_postfix}.{target_fex}")
    mask = read_image(msk_dir, ImageReadMode.GRAY)
    return mask


# gathering images
images = []
masks = []
for ID in tqdm(data_ids):
    img = get_img_by_id(ID)
    msk = get_msk_by_id(ID)

    if img_transform:
        img = img_transform(img)
        img = (img - img.min()) / (img.max() - img.min())
    if msk_transform:
        msk = msk_transform(msk)
        msk = (msk - msk.min()) / (msk.max() - msk.min())

    img = img.numpy()
    msk = msk.numpy()

    images.append(img)
    masks.append(msk)

X = np.array(images)
Y = np.array(masks)

np.save("X_tr_224x224", X)
np.save("Y_tr_224x224", Y)
