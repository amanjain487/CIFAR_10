# CIFAR_10
Personal practice on CIFAR10 with PyTorch

## Introduction
- The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

## Contents
1. main.py
  - training and test loops
  - data splits between test and train
  - train data for specified number of epochs
  - which optimizer to run
  - which loss function to use
  - do we need a scheduler
2. utils.py
  - image transforms
  - display images, ghraphs
  - tensorboard related stuff
  - advanced training policie
3. GradCAM
  - Files which have required functions for gradcam.
    - gradcam.py
    - visualize.py
4. models
  - Collection of different models for training.
    - resnet.py
  
