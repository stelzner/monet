import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import visdom

import model
import datasets

vis = visdom.Visdom()

class MonetArgs:
    def __init__(self):
        self.vis_every = 50
        self.load_parameters = True
        self.checkpoint_file = './checkpoints/monet.ckpt'

def numpify(tensor):
    return tensor.cpu().detach().numpy()

def visualize_masks(imgs, masks, recons):
    print('recons min/max', recons.min().item(), recons.max().item())
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    colors.extend([(c[0]//2, c[1]//2, c[2]//2) for c in colors])
    colors.extend([(c[0]//4, c[1]//4, c[2]//4) for c in colors])
    seg_maps = np.zeros_like(imgs)
    masks = np.argmax(masks, 1)
    for i in range(imgs.shape[0]):
        for y in range(imgs.shape[2]):
            for x in range(imgs.shape[3]):
                seg_maps[i, :, y, x] = colors[masks[i, y, x]]

    seg_maps /= 255.0
    vis.images(np.concatenate((imgs, seg_maps, recons), 0), nrow=imgs.shape[0])


if __name__ == '__main__':
    args = MonetArgs()
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float()),
                                    ])
    trainset = datasets.Sprites(train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    monet = model.Monet().cuda()

    if args.load_parameters:
        monet.load_state_dict(torch.load(args.checkpoint_file))
    else:
        for w in monet.parameters():
            # print(w.shape)
            # print('before', w.min(), w.max())
            std_init = 0.01
            nn.init.normal_(w, mean=0., std=std_init)
            # print('after', w.min(), w.max())

    optimizer = optim.RMSprop(monet.parameters(), lr=1e-4)

    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, counts = data
            images = images.cuda()
            optimizer.zero_grad()
            loss = torch.mean(monet(images))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % args.vis_every == args.vis_every-1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / args.vis_every))
                running_loss = 0.0
                visualize_masks(numpify(images[:8]),
                                numpify(monet.masks[:8]),
                                numpify(monet.complete_recon[:8]))

        torch.save(monet.state_dict(), args.checkpoint_file)

    print('done')
