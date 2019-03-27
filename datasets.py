import os
import sys

import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.random import random_integers


def progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def make_sprites(n=50000, height=64, width=64):
    images = np.zeros((n, height, width, 3))
    counts = np.zeros((n,))
    print('Generating sprite dataset...')
    for i in range(n):
        num_sprites = random_integers(0, 2)
        counts[i] = num_sprites
        for j in range(num_sprites):
            pos_y = random_integers(0, height - 12)
            pos_x = random_integers(0, width - 12)

            scale = random_integers(12, min(16, height-pos_y, width-pos_x))

            cat = random_integers(0, 2)
            sprite = np.zeros((height, width, 3))

            if cat == 0:  # draw circle
                center_x = pos_x + scale // 2.0
                center_y = pos_y + scale // 2.0
                for x in range(height):
                    for y in range(width):
                        dist_center_sq = (x - center_x)**2 + (y - center_y)**2
                        if  dist_center_sq < (scale // 2.0)**2:
                            sprite[x][y][cat] = 1.0
            elif cat == 1:  # draw square
                sprite[pos_x:pos_x + scale, pos_y:pos_y + scale, cat] = 1.0
            else:  # draw square turned by 45 degrees
                center_x = pos_x + scale // 2.0
                center_y = pos_y + scale // 2.0
                for x in range(height):
                    for y in range(width):
                        if abs(x - center_x) + abs(y - center_y) < (scale // 2.0):
                            sprite[x][y][cat] = 1.0
            images[i] += sprite
        if i % 100 == 0:
            progress_bar(i, n)
            # print('{}/{}'.format(i, n))
    # print('{}/{}'.format(n, n))
    images = np.clip(images, 0.0, 1.0)

    return {'x_train': images[:4 * n // 5],
            'count_train': counts[:4 * n // 5],
            'x_test': images[4 * n // 5:],
            'count_test': counts[4 * n // 5:]}


class Sprites(Dataset):
    def __init__(self, n=50000, canvas_size=64, np_file=None, train=True, transform=None):
        if np_file is None:
            np_file = './data/sprites_{}_{}.npz'.format(n, canvas_size)
            if not os.path.isfile(np_file):
                gen_data = make_sprites(n, canvas_size, canvas_size)
                np.savez(np_file, **gen_data)

        data = np.load(np_file)

        self.transform = transform
        self.images = data['x_train'] if train else data['x_test']
        # self.images = np.transpose(self.images, (0, 3, 1, 2))
        self.counts = data['count_train'] if train else data['count_test']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.counts[idx]

