import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic
    parser.add_argument('--name', default = 'main', help = '')

    # For Loader
    parser.add_argument('--path', default = 'data', help = '')

    # For Module
    parser.add_argument('--dis_layers_size', type = list, default = [784, 512, 256, 1], help = '')
    parser.add_argument('--gen_layers_size', type = list, default = [256, 512, 784], help = '')
    parser.add_argument('--gen_latent_size', type = int, default = 64, help = '')

    # For Train
    parser.add_argument('--batch_size', type = int, default = 64, help = '')
    parser.add_argument('--num_epochs', type = int, default = 10, help = '')

    return parser
