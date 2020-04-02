# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision
import torchvision.transforms as transforms
# import numpy as np

import torch.nn as nn
# import torch.nn.functional as F

import torch.optim as optim

from model import Net
from util import imshow, classes

def train():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )

    # # show
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()

    # imshow(torchvision.utils.make_grid(images))

    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    # model
    net = Net()
    net.to(device)

    # loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            # print('outputs: ', outputs)
            # print('labels: ', labels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    savepath = './cifar_net_epoch10.pth'
    torch.save(net.state_dict(), savepath)

if __name__ == '__main__':
    train()