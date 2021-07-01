import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

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
  if "random_rotate" in train:
    train_list.append(transforms.RandomRotation((-5.0, 5.0), fill=(0,0,0)))
  if "cutout" in train:
    albumentation_train_list.append(A.CoarseDropout(p=0.5, max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value = None))
  if "shift_scale_rotate" in train:
     albumentation_train_list.append(A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5))
  if "grayscale" in train:
     albumentation_train_list.append(A.ToGray(p=0.5))
  
  return transforms.Compose(train_list), A.Compose(albumentation_train_list)



def load_dataset(tensor_train, numpy_train):
  train_dataset = CIFAR_10_Dataset(torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                  transform=tensor_train), numpy_train)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                                                   ]))
  return train_dataset, testset


def plot_graph(tr_l, tr_a, te_l, te_a):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(tr_l)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(tr_a)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(te_l)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(te_a)
  axs[1, 1].set_title("Test Accuracy")


def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    single_img = False
    if tensor.ndimension() == 3:
      single_img = True
      tensor = tensor[None,:,:,:]

    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    ret = tensor.mul(std).add(mean)
    return ret[0] if single_img else ret

  
def identify_images(net, criterion, device, testloader):
    net.eval()
    correct_images = []
    incorrect_images = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)           
            predicted = outputs.argmax(dim=1, keepdim=True)
            is_correct = predicted.eq(targets.view_as(predicted))
            
            misclassified_inds = (is_correct==0).nonzero()[:,0]
            for mis_ind in misclassified_inds:
              if len(incorrect_images) == 25:
                break
              incorrect_images.append({
                  "target": targets[mis_ind].cpu().numpy(),
                  "pred": predicted[mis_ind][0].cpu().numpy(),
                  "img": inputs[mis_ind]
              })

            correct_inds = (is_correct==1).nonzero()[:,0]
            for ind in correct_inds:
              if len(correct_images) == 25:
                break
              correct_images.append({
                  "target": targets[ind].cpu().numpy(),
                  "pred": predicted[ind][0].cpu().numpy(),
                  "img": inputs[ind]
              })
    return correct_images, incorrect_images
  
  
def plot_images(img_data, classes):
    figure = plt.figure(figsize=(10, 10))

    num_of_images = len(img_data)
    for index in range(1, num_of_images + 1):
        img = denormalize(img_data[index-1]["img"])  # unnormalize
        plt.subplot(5, 5, index)
        plt.axis('off')
        plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
        plt.title("Predicted: %s\nActual: %s" % (classes[img_data[index-1]["pred"]], classes[img_data[index-1]["target"]]))

    plt.tight_layout()
