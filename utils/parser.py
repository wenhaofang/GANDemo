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
    parser.add_argument('--module', type = int, choices = range(1, 4), default = 1, help = '')

    parser.add_argument('--batch_size', type = int, default = 128, help = '')
    parser.add_argument('--num_epochs', type = int, default = 100, help = '')

    parser.add_argument('--num_labels', type = int, default = 10, help = '')

    parser.add_argument('--saved_s_freq', type = int, default = 1, help = '')
    parser.add_argument('--saved_m_freq', type = int, default = 5, help = '')

    parser.add_argument('--train_d_freq', type = int, default = 1, help = '')
    parser.add_argument('--train_g_freq', type = int, default = 1, help = '')

    parser.add_argument('--lr', type = float, default = 0.0002, help = '')

    parser.add_argument('--clip_params', type = float, default = 0.01, help = '')

    parser.add_argument('--weight_gp', type = float, default = 10, help = '')

    return parser
