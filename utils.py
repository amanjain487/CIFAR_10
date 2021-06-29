import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import albumentations as A


class CIFAR_10_Dataset(torch.utils.data.Dataset):
  def __init__(self,  dataset, transformer=None):
        self.dataset = dataset
        self.transforms = transformer
  def __len__(self):
        return len(self.dataset)
  def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img, target = self.dataset[idx]
        img = img.cpu().detach().numpy()
        img = np.asarray(img).reshape((32,32,3))
        if self.transforms is not None:
            image = self.transforms(image=img)
        img = torch.from_numpy(img.reshape(3,32,32))
        return img, target


def train_transform(train):
  albumentation_train_list = []
  train_list = []
  if "totensor" in train:
    train_list.append(transforms.ToTensor())
  if "normalize_normal" in train:
    train_list.append(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
  if "normalize_mean" in train:
    train_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
  if "randomcrop" in train:
    train_list.append(transforms.RandomCrop(32, padding=4))
  if "horizontal_flip" in train:
    train_list.append(transforms.RandomHorizontalFlip())
  if "cutout" in train:
    albumentation_train_list.append(A.CoarseDropout(p=0.5, max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value = None))
  if "shift_scale_rotate" in train:
     albumentation_train_list.append(A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5))
  if "grayscale" in train:
     albumentation_train_list.append(A.ToGray(p=0.5))
  
  return transforms.Compose(train_list), A.Compose(albumentation_train_list)



def load_dataset(tensor_train, numpy_train):
  train_dataset = CIFAR_10_Dataset(datasets.CIFAR10(root='./data', train=True, download=True,
                                  transform=tensor_train), numpy_train)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                                                   ]))
  return train_dataset, testset


