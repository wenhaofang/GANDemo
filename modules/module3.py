# WGAN-GP | the same as WGAN

import torch
import torch.nn as nn

class Dis(nn.Module):
    def __init__(
        self,
        layers_size: list
    ):
        super(Dis, self).__init__()

        self.dis = nn.Sequential()

        # diff: 不能使用 Batch Normalization
        for layer_i, (
            i_size,
            o_size,
        ) in enumerate(zip(
            layers_size[:-1],
            layers_size[+1:],
        )):
            self.dis.add_module(
                name = 'd_l_{}'.format(layer_i), module = nn.Linear(i_size, o_size)
            )
            if  layer_i + 2 < len(layers_size):
                self.dis.add_module(
                    name = 'd_a_{}'.format(layer_i), module = nn.LeakyReLU(0.2)
                )

    def forward(self,x):

        y = self.dis(x).squeeze()

        return y

class Gen(nn.Module):
    def __init__(
        self,
        layers_size: list,
        latent_size: int,
    ):
        super(Gen, self).__init__()

        self.gen = nn.Sequential()

        for layer_i, (
            i_size,
            o_size,
        ) in enumerate(zip(
            [latent_size] + layers_size[:-1],
            (layers_size)
        )):
            self.gen.add_module(
                name = 'g_l_{}'.format(layer_i), module = nn.Linear(i_size, o_size)
            )
            self.gen.add_module(
                name = 'g_a_{}'.format(layer_i), module = nn.ReLU() if layer_i + 1 < len(layers_size) else nn.Tanh()
            )

    def forward(self,z):

        x = self.gen(z)

        return x

class GAN(nn.Module):
    def __init__(
        self,
        dis_layers_size,
        gen_layers_size,
        gen_latent_size,
    ):
        super(GAN, self).__init__()
        self.dis = Dis(dis_layers_size)
        self.gen = Gen(gen_layers_size, gen_latent_size)

def get_module(option):
    return GAN(
        option.dis_layers_size,
        option.gen_layers_size,
        option.gen_latent_size,
    )

if  __name__ == '__main__':

    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    module = get_module(option)

    channel = 1
    image_w = 28
    image_h = 28

    x = torch.randn((option.batch_size, channel * image_w * image_h))

    z = torch.randn((option.batch_size, option.gen_latent_size))

    y1 = module.dis(x)

    y2 = module.gen(z)

    print(y1.shape) # (batch_size)
    print(y2.shape) # (batch_size, channel * image_w * image_h)
