import torch
from torch import nn, optim, autograd
import numpy as np
import visdom
from torch.nn import functional as F
from matplotlib import pyplot as plt
import random

h_dim = 400
batchsz = 512
noise_size = 4
feature_size = 4
viz = visdom.Visdom()


def data_generator():
    center = np.array([50, 50], dtype=np.float32)

    while True:
        dataset = []
        for i in range(batchsz):
            size = np.random.uniform(0, 50, 2).astype(np.float32)
            # size = np.random.randn(2).astype(np.float32)
            lt = (center - size) / 100
            rb = (center + size) / 100
            dataset.append(np.concatenate((lt, rb), axis=0))
        dataset = np.array(dataset)
        # max_val = dataset.max()
        # dataset = dataset / max_val
        yield dataset


# def data_generator():
#     scale = 2.
#     centers = [
#         (1, 0),
#         (-1, 0),
#         (0, 1),
#         (0, -1),
#         (1. / np.sqrt(2), 1. / np.sqrt(2)),
#         (1. / np.sqrt(2), -1. / np.sqrt(2)),
#         (-1. / np.sqrt(2), 1. / np.sqrt(2)),
#         (-1. / np.sqrt(2), -1. / np.sqrt(2))
#     ]
#     centers = [(scale * x, scale * y) for x, y in centers]
#     while True:
#         dataset = []
#         for i in range(batchsz):
#             point1 = np.random.randn(2) * .02
#             center = random.choice(centers)
#             point1[0] += center[0]
#             point1[1] += center[1]
#
#             point2 = np.random.randn(2) * .02
#             center = random.choice(centers)
#             point2[0] += center[0]
#             point2[1] += center[1]
#
#             dataset.append(np.concatenate((point1, point2), axis=0))
#         dataset = np.array(dataset, dtype='float32')
#         dataset /= 1.414  # stdev
#         yield dataset


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(feature_size, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, feature_size),
        )

    def forward(self, z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(feature_size, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)


def weights_init(m):
    if isinstance(m, nn.Linear):
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def gradient_penalty(D, xr, xf):
    LAMBDA = 0.3

    # only constrait for Discriminator
    xf = xf.detach()
    xr = xr.detach()

    # [b, 1] => [b, 2]
    alpha = torch.rand(batchsz, 1).cuda()
    alpha = alpha.expand_as(xr)

    interpolates = alpha * xr + ((1 - alpha) * xf)
    interpolates.requires_grad_()

    disc_interpolates = D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gp


if __name__ == '__main__':
    torch.manual_seed(23)
    np.random.seed(23)

    G = Generator().cuda()
    D = Discriminator().cuda()
    G.apply(weights_init)
    D.apply(weights_init)

    optim_G = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.9))

    data_iter = data_generator()
    print('batch:', next(data_iter).shape)

    for epoch in range(50000):

        # 1. train discriminator for k steps
        for _ in range(5):
            x = next(data_iter)
            xr = torch.from_numpy(x).cuda()

            # [b]
            predr = (D(xr))
            # max log(lossr)
            lossr = - (predr.mean())

            # [b, 2]
            z = torch.randn(batchsz, noise_size).cuda()
            # stop gradient on G
            # [b, 2]
            xf = G(z).detach()
            # [b]
            predf = (D(xf))
            # min predf
            lossf = (predf.mean())

            # gradient penalty
            gp = gradient_penalty(D, xr, xf)

            loss_D = lossr + lossf + gp
            optim_D.zero_grad()
            loss_D.backward()
            # for p in D.parameters():
            #     print(p.grad.norm())
            optim_D.step()

        # 2. train Generator
        z = torch.randn(batchsz, noise_size).cuda()
        xf = G(z)
        predf = (D(xf))
        # max predf
        loss_G = - (predf.mean())
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
            viz.line([[loss_D.item(), loss_G.item(), gp.item()]],
                     [[epoch, epoch, epoch]],
                     win='loss',
                     update='append',
                     opts={"title": "loss", "legend": ["loss d", "loss g", "gp"]}
                     )

            # generate_image(D, G, xr, epoch)

            print(loss_D.item(), loss_G.item(), gp.item())
