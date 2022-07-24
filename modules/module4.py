# CGAN

import torch
import torch.nn as nn

class Dis(nn.Module):
    def __init__(
        self,
        layers_size: list,
        num_labels : int,
    ):
        super(Dis, self).__init__()

        self.dis = nn.Sequential()

        for layer_i, (
            i_size,
            o_size,
        ) in enumerate(zip(
            layers_size[:-1],
            layers_size[+1:],
        )):
            if  layer_i == 0:
                i_size += num_labels # diff

            self.dis.add_module(
                name = 'd_l_{}'.format(layer_i), module = nn.Linear(i_size, o_size)
            )
            self.dis.add_module(
                name = 'd_a_{}'.format(layer_i), module = nn.LeakyReLU(0.2)
            )

        self.dis.add_module(
            name = 'd_l_{}'.format(len(layers_size) - 1), module = nn.Linear (layers_size[-1], 1)
        )
        self.dis.add_module(
            name = 'd_a_{}'.format(len(layers_size) - 1), module = nn.Sigmoid() # Softmax?
        )

    def forward(self,x,c):

        x = torch.cat((x, c), dim = -1) # diff

        y = self.dis(x).squeeze()

        return y

class Gen(nn.Module):
    def __init__(
        self,
        layers_size: list,
        latent_size: int,
        num_labels : int,
    ):
        super(Gen, self).__init__()

        self.gen = nn.Sequential()

        for layer_i, (
            i_size,
            o_size,
        ) in enumerate(zip(
            [latent_size + num_labels] + layers_size[:-1], # diff
            (layers_size)
        )):
            self.gen.add_module(
                name = 'g_l_{}'.format(layer_i), module = nn.Linear(i_size, o_size)
            )
            self.gen.add_module(
                name = 'g_a_{}'.format(layer_i), module = nn.ReLU() if layer_i + 1 < len(layers_size) else nn.Tanh()
            )

    def forward(self,z,c):

        z = torch.cat((z, c), dim = -1) # diff

        x = self.gen(z).squeeze()

        return x

class GAN(nn.Module):
    def __init__(
        self,
        dis_layers_size,
        gen_layers_size,
        gen_latent_size,
        num_labels,
    ):
        super(GAN, self).__init__()
        self.dis = Dis(dis_layers_size, num_labels)
        self.gen = Gen(gen_layers_size, gen_latent_size, num_labels)

def get_module(option):
    return GAN(
        option.dis_layers_size,
        option.gen_layers_size,
        option.gen_latent_size,
        option.num_labels,
    )

if  __name__ == '__main__':

    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    module = get_module(option)

    channel = 1
    image_w = 28
    image_h = 28

    from utils.misc import to_onehot

    x = torch.randn((option.batch_size, channel * image_w * image_h))

    z = torch.randn((option.batch_size, option.gen_latent_size))

    c = to_onehot(torch.randint(option.num_labels, (option.batch_size,)), option.num_labels)

    y1 = module.dis(x, c)

    y2 = module.gen(z, c)

    print(y1.shape) # (batch_size)
    print(y2.shape) # (batch_size, channel * image_w * image_h)
