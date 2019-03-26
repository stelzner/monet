import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import model
import datasets

if __name__ == '__main__':
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float())])
    trainset = datasets.Sprites('./data/sprites_10000_50.npz', train=True,
                               transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    monet = model.Monet().cuda()
    for w in monet.parameters():
        nn.init.normal_(w)
    optimizer = optim.RMSprop(monet.parameters(), lr=1e-4)

    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, counts = data
            print(images.shape, images.dtype)
            images = images.cuda()
            optimizer.zero_grad()
            loss = torch.sum(monet(images))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('done')
