import torch
import torch.nn as nn
import torch.distributions as dists
import pyro

import torchvision


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, channel_base=64):
        super().__init__()
        self.num_blocks = num_blocks
        self.down_convs = nn.ModuleList()
        cur_in_channels = in_channels
        for i in range(num_blocks):
            self.down_convs.append(double_conv(cur_in_channels,
                                               channel_base * 2**i))
            cur_in_channels = channel_base * 2**i

        self.tconvs = nn.ModuleList()
        for i in range(num_blocks-1, 0, -1):
            self.tconvs.append(nn.ConvTranspose2d(channel_base * 2**i,
                                                  channel_base * 2**(i-1),
                                                  2, stride=2))

        self.up_convs = nn.ModuleList()
        for i in range(num_blocks-2, -1, -1):
            self.up_convs.append(double_conv(channel_base * 2**(i+1), channel_base * 2**i))

        self.final_conv = nn.Conv2d(channel_base, out_channels, 1)

    def forward(self, x):
        intermediates = []
        cur = x
        for down_conv in self.down_convs[:-1]:
            cur = down_conv(cur)
            intermediates.append(cur)
            cur = nn.MaxPool2d(2)(cur)

        cur = self.down_convs[-1](cur)

        for i in range(self.num_blocks-1):
            cur = self.tconvs[i](cur)
            cur = torch.cat((cur, intermediates[-i -1]), 1)
            cur = self.up_convs[i](cur)

        return self.final_conv(cur)


class AttentionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(num_blocks=5, in_channels=4, out_channels=2)

    def forward(self, x, scope):
        inp = torch.cat((x, scope), 1)
        logits = self.unet(inp)
        log_alpha = torch.softmax(logits, 1)
        # output channel 0 represents log alpha_k,
        # channel 1 represents log (1 - alpha_k).
        mask = scope * log_alpha[:, 0:1]
        new_scope = scope * log_alpha[:, 1:2]
        return mask, new_scope

class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True)
        )
        self.mlp = nn.Sequential(
            nn.Linear(64 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x

class DecoderNet(nn.Module):
    def __init__(self, img_width=64, img_height=64):
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.convs = nn.Sequential(
            nn.Conv2d(18, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1),
            nn.Sigmoid()
        )
        ys = torch.linspace(-1, 1, self.img_height+8)
        xs = torch.linspace(-1, 1, self.img_width+8)
        ys, xs = torch.meshgrid(ys, xs)
        self.coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', self.coord_map)

    def forward(self, z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.img_height+8, self.img_width+8)
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.convs(inp)
        return result


class Monet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = AttentionNet()
        self.encoder = EncoderNet()
        self.decoder = DecoderNet()
        self.k = 4
        self.beta = 0.5
        self.gamma = 0.25
        self.masks = None
        self.complete_recon = None

    def forward(self, x):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        for i in range(self.k-1):
            mask, scope = self.attention(x, scope)
            masks.append(mask)
        masks.append(scope)
        loss = torch.zeros_like(x[:, 0, 0, 0])
        mask_preds = []
        complete_recon = torch.zeros_like(x)
        for i, mask in enumerate(masks):
            z, kl_z = self.__encoder_step(x, mask)
            sigma = 0.09 if i == 0 else 0.11
            p_x, x_recon, mask_pred = self.__decoder_step(x, z, mask, sigma)
            mask_preds.append(mask_pred)
            # print('px', p_x.mean().item())
            # print('klz', kl_z.mean().item())
            loss += -p_x + self.beta * kl_z
            complete_recon += mask * x_recon

        self.masks = torch.cat(masks, 1)
        self.complete_recon = complete_recon
        masks = torch.transpose(self.masks, 1, 3)
        q_masks = dists.Categorical(logits=masks)
        q_masks_recon = dists.Categorical(logits=torch.stack(mask_preds, 3))
        kl_masks = dists.kl_divergence(q_masks_recon, q_masks)
        kl_masks = torch.sum(kl_masks, [1, 2])
        # print('kl masks', kl_masks.mean().item())
        loss += self.gamma * kl_masks
        return loss


    def __encoder_step(self, x, mask):
        encoder_input = torch.cat((x, mask), 1)
        q_params = self.encoder(encoder_input)
        means = torch.sigmoid(q_params[:, :16]) * 6 - 3
        sigmas = torch.sigmoid(q_params[:, 16:]) * 3
        dist = dists.Normal(means, sigmas)
        dist_0 = dists.Normal(0., sigmas)
        z = means + dist_0.sample()
        # print('z', z.min().item(), z.max().item())
        q_z = dist.log_prob(z)
        # print('means', means)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        # print('px dists', dist)
        # print('kl_z', kl_z)
        kl_z = torch.sum(kl_z, 1)
        return z, kl_z

    def __decoder_step(self, x, z, mask, sigma):
        decoder_output = self.decoder(z)
        x_recon = decoder_output[:, :3]
        # print('recon', x_recon.min().item(), x_recon.max().item())
        mask_pred = decoder_output[:, 3]
        dist = dists.Normal(x_recon, sigma)
        p_x = dist.log_prob(x)
        # print('px min/max', p_x.min().item(), p_x.max().item())
        # print(mask.shape, p_x.shape)
        p_x *= mask
        p_x = torch.sum(p_x, [1, 2, 3])
        return p_x, x_recon, mask_pred





