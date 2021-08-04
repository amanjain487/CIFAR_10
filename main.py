import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import os
import argparse


# Training
def train(epoch, net, criterion, optimizer, device, trainloader, train_losses, train_acc):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_losses.append(train_loss/(batch_idx+1))
    train_acc.append(100.*correct/total)
    return train_losses, train_acc


def test(epoch, net, criterion, device, testloader, best_acc, test_losses, test_acc):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    test_losses.append(test_loss/(batch_idx+1))
    test_acc.append(100.*correct/total)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    return best_acc, test_losses, test_acc
    


def dataloaders(trainset, testset):
    trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)
    
    return trainloader, testloader


def start_training(no_of_epoch, net, criterion, optimizer, device, trainloader, testloader, best_acc, scheduler):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    
    for epoch in range(no_of_epoch):
        train_loss, train_acc = train(epoch+1, net, criterion, optimizer, device, trainloader, train_loss, train_acc)
        best_acc, test_loss, test_acc = test(epoch+1, net, criterion, device, testloader, best_acc, test_loss, test_acc)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss[-1])
    print("Best Acc is : ", best_acc)
    return train_loss, train_acc, test_loss, test_acc

        
def define_model_utilities(model, momentum, weight_decay, scheduler, loss="cross_entropy", optimizer_func="SGD", lr=0.1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

    net = model
    net = net.to(device)
     
    if loss=="cross_entropy":
        criterion = nn.CrossEntropyLoss()
    
    if optimizer_func=="SGD":
        optimizer = optim.SGD(net.parameters(), lr=lr,
                      momentum=momentum)

    if scheduler is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    return device, best_acc, classes, net, criterion, optimizer, scheduler
