from torch.utils.data.dataset import Dataset
from PIL import Image
from PIL import ImageFilter
import random
import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
import torch
import numpy as np
from glob import glob
import os
import pandas as pd

# --------------------------------------------------------------------------------
# Define data augmentation
# --------------------------------------------------------------------------------
def transform(image, label=None, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, Image.BILINEAR)
    if label is not None:
        label = transforms_f.resize(label, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, Image.NEAREST)

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        if label is not None:
            label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    if label is not None:
        label = transforms_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transforms_f.crop(logits, i, j, h, w)

    if augmentation:
        # Random color jitter
        if torch.rand(1) > 0.2:
            color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))  # For PyTorch 1.9/TorchVision 0.10 users
            # color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)

        # Random Gaussian filter
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            if label is not None:
                label = transforms_f.hflip(label)
            if logits is not None:
                logits = transforms_f.hflip(logits)

    # Transform to tensor
    image = transforms_f.to_tensor(image)
    if label is not None:
        label = (transforms_f.to_tensor(label) * 255).long()
        label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)

    # Apply (ImageNet) normalisation
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits is not None:
        return image, label, logits
    else:
        return image, label

def denormalise(x, imagenet=True):
    if imagenet:
        x = transforms_f.normalize(x, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        return x
    else:
        return (x + 1) / 2

def tensor_to_pil(im, label, logits):
    im = denormalise(im)
    im = transforms_f.to_pil_image(im.cpu())

    label = label.float() / 255.
    label = transforms_f.to_pil_image(label.unsqueeze(0).cpu())

    logits = transforms_f.to_pil_image(logits.unsqueeze(0).cpu())
    return im, label, logits

def batch_transform(data, label, logits, crop_size, scale_size, apply_augmentation):
    data_list, label_list, logits_list = [], [], []
    device = data.device

    for k in range(data.shape[0]):
        data_pil, label_pil, logits_pil = tensor_to_pil(data[k], label[k], logits[k])
        aug_data, aug_label, aug_logits = transform(data_pil, label_pil, logits_pil,
                                                    crop_size=crop_size,
                                                    scale_size=scale_size,
                                                    augmentation=apply_augmentation)
        data_list.append(aug_data.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)

    data_trans, label_trans, logits_trans = \
        torch.cat(data_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device)
    return data_trans, label_trans, logits_trans


# --------------------------------------------------------------------------------
# Define indices for labelled, unlabelled training images, and test images
# --------------------------------------------------------------------------------
def get_harbor_idx(root, train=True, is_label=True ,label_num=15):
    if train:
        if is_label:
            classes = ['ship', 'container_truck', 'forklift', 'reach_stacker']
            image_path = glob(os.path.join(root, 'train', 'labeled_images', '*.jpg'))
            image_idx_list = list(map(lambda x : x.split('\\')[-1].split('.')[0], image_path))
            train_idx = []
            valid_idx = []
            for c in classes:
                matched_idx = [i for i in image_idx_list if c in i]
                train_idx.extend(matched_idx[:label_num])
                valid_idx.extend(matched_idx[label_num:])
            return train_idx, valid_idx
        else:
            image_path = glob(os.path.join(root, 'train', 'unlabeled_images', '*.jpg'))
            train_idx = list(map(lambda x: x.split('\\')[-1].split('.')[0], image_path))
            return train_idx
    else:
        image_path = glob(os.path.join(root, 'test', 'images', '*.jpg'))
        test_idx = list(map(lambda x: x.split('/')[-1].split('.')[0], image_path))
        return test_idx

# --------------------------------------------------------------------------------
# Create dataset in PyTorch format
# --------------------------------------------------------------------------------
class BuildDataset(Dataset):
    def __init__(self, root, idx_list, crop_size=(512, 512), scale_size=(0.5, 2.0),
                 augmentation=True, train=True, is_label=True):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.idx_list = idx_list
        self.scale_size = scale_size
        self.is_label = is_label

    def __getitem__(self, index):
        if self.train:
            if self.is_label:
                image_root = Image.open(self.root + f'/train/labeled_images/{self.idx_list[index]}.jpg')
                label_root = Image.open(self.root + f'/train/labels/{self.idx_list[index]}.png')
            else:
                image_root = Image.open(self.root + f'/train/unlabeled_images/{self.idx_list[index]}.jpg')
                label_root = None

            image, label = transform(image_root, label_root, None, self.crop_size, self.scale_size, self.augmentation)
            if label is not None:
                return image, label.squeeze(0)
            else:
                return image

        else:
            file_name = f'{self.idx_list[index]}.jpg'
            image_root = Image.open(self.root + f'/test/images/{file_name}')
            image, label = transform(image_root, None, None, self.crop_size, self.scale_size, self.augmentation)
            return image, torch.tensor(image_root.size), file_name

    def __len__(self):
        return len(self.idx_list)


# --------------------------------------------------------------------------------
# Create data loader in PyTorch format
# --------------------------------------------------------------------------------
class BuildDataLoader:
    def __init__(self, num_labels, dataset_path, batch_size):
        self.data_path = dataset_path
        self.im_size = [513, 513]
        self.crop_size = [321, 321]
        self.num_segments = 5
        self.scale_size = (0.5, 1.5)
        self.batch_size = batch_size
        self.train_l_idx, self.valid_l_idx = get_harbor_idx(self.data_path, train=True, is_label=True, label_num=num_labels)
        self.train_u_idx = get_harbor_idx(self.data_path, train=True, is_label=False)
        self.test_idx = get_harbor_idx(self.data_path, train=False)

        if num_labels == 0:  # using all data
            self.train_l_idx = self.train_u_idx

    def build(self, supervised=False):
        train_l_dataset = BuildDataset(self.data_path, self.train_l_idx,
                                       crop_size=self.crop_size, scale_size=self.scale_size,
                                       augmentation=True, train=True, is_label=True)
        train_u_dataset = BuildDataset(self.data_path, self.train_u_idx,
                                       crop_size=self.crop_size, scale_size=(1.0, 1.0),
                                       augmentation=False, train=True, is_label=False)
        valid_l_dataset = BuildDataset(self.data_path, self.valid_l_idx,
                                       crop_size=self.crop_size, scale_size=self.scale_size,
                                       augmentation=False, train=True, is_label=True)
        test_dataset    = BuildDataset(self.data_path, self.test_idx,
                                       crop_size=self.im_size, scale_size=(1.0, 1.0),
                                       augmentation=False, train=False, is_label=True)

        if supervised:  # no unlabelled dataset needed, double batch-size to match the same number of training samples
            self.batch_size = self.batch_size * 2

        num_samples = self.batch_size * 200  # for total 40k iterations with 200 epochs
        # num_samples = self.batch_size * 2
        train_l_loader = torch.utils.data.DataLoader(
            train_l_dataset,
            batch_size=self.batch_size,
            sampler=sampler.RandomSampler(data_source=train_l_dataset,
                                          replacement=True,
                                          num_samples=num_samples),
            drop_last=True,)


        valid_l_loader = torch.utils.data.DataLoader(
            valid_l_dataset,
            batch_size=self.batch_size,
            sampler=sampler.RandomSampler(data_source=valid_l_dataset,
                                          replacement=True,
                                          num_samples=num_samples),
            drop_last=True,)

        if not supervised:
            train_u_loader = torch.utils.data.DataLoader(
                train_u_dataset,
                batch_size=self.batch_size,
                sampler=sampler.RandomSampler(data_source=train_u_dataset,
                                              replacement=True,
                                              num_samples=num_samples),
                drop_last=True,)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        if supervised:
            return train_l_loader, valid_l_loader, test_loader
        else:
            return train_l_loader, train_u_loader, valid_l_loader, test_loader

# --------------------------------------------------------------------------------
# Create Color-mapping for visualisation
# --------------------------------------------------------------------------------

def color_map(mask, colormap):
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]
    return np.uint8(color_mask)

