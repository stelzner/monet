import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import visdom

import os

import model
import datasets

vis = visdom.Visdom()

class MonetArgs:
    def __init__(self):
        self.vis_every = 50
        self.num_slots = 4
        self.load_parameters = True
        self.checkpoint_file = './checkpoints/monet.ckpt'
        self.batch_size = 64
        self.num_epochs = 20
        self.num_blocks = 5
        self.channel_base = 64

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

def run_training(monet, args, trainloader):
    if args.load_parameters and os.path.isfile(args.checkpoint_file):
        monet.load_state_dict(torch.load(args.checkpoint_file))
        print('Restored parameters from', args.checkpoint_file)
    else:
        for w in monet.parameters():
            std_init = 0.01
            nn.init.normal_(w, mean=0., std=std_init)
        print('Initialized parameters')

    optimizer = optim.RMSprop(monet.parameters(), lr=1e-4)

    for epoch in range(args.num_epochs):
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

    print('training done')

def sprite_experiment():
    args = MonetArgs()
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float()),
                                    ])
    trainset = datasets.Sprites(train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    monet = model.Monet(args, 64, 64).cuda()
    run_training(monet, args, trainloader)

def clevr_experiment():
    args = MonetArgs()
    args.channel_base = 16
    args.batch_size = 20
    args.num_slots = 11
    args.num_blocks = 6
    args.checkpoint_file = './checkpoints/clevr.ckpt'
    # Crop as described in appendix C
    crop_tf = transforms.Lambda(lambda x: transforms.functional.crop(x, 29, 64, 192, 192))
    drop_alpha_tf = transforms.Lambda(lambda x: x[:3])
    transform = transforms.Compose([crop_tf,
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    drop_alpha_tf,
                                    transforms.Lambda(lambda x: x.float()),
                                   ])
    trainset = datasets.Clevr('/data/stelzner/data/CLEVR_v1.0/images/train',
                              transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)
    monet = model.Monet(args, 128, 128).cuda()
    run_training(monet, args, trainloader)

if __name__ == '__main__':
    clevr_experiment()
    # sprite_experiment()

