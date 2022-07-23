import os
import subprocess

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from utils.parser import get_parser
from utils.logger import get_logger

parser = get_parser()
option = parser.parse_args()

root_path = 'result'

logs_folder = os.path.join(root_path, 'logs', option.name)
save_folder = os.path.join(root_path, 'save', option.name)
sample_folder = os.path.join(root_path, 'sample', option.name)
result_folder = os.path.join(root_path, 'result', option.name)

subprocess.run('mkdir -p %s' % logs_folder, shell = True)
subprocess.run('mkdir -p %s' % save_folder, shell = True)
subprocess.run('mkdir -p %s' % sample_folder, shell = True)
subprocess.run('mkdir -p %s' % result_folder, shell = True)

logs_path = os.path.join(logs_folder, 'main.log')
save_path = os.path.join(save_folder, 'best.pth')

logger = get_logger(option.name, logs_path)

from loaders.loader1 import get_loader as get_loader1 # MNIST

from modules.module1 import get_module as get_module1 # GAN

from utils.misc import train_d, train_g, save_checkpoint, load_checkpoint, save_sample

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logger.info('prepare loader')

loader = get_loader1(option)

logger.info('prepare module')

module = get_module1(option).to(device) if option.module == 1 else \
         None

logger.info('prepare envs')

optimizerD = optim.Adam(module.dis.parameters(), lr = option.learning_rate)
optimizerG = optim.Adam(module.gen.parameters(), lr = option.learning_rate)

criterion = nn.BCELoss()

logger.info('start training!')

for epoch in range(1, 1 + option.num_epochs):

    print_a_fstr = '[Epoch {:d}]'.format(epoch)
    train_d_fstr = ''
    train_g_fstr = ''

    if  epoch % option.train_d_freq == 0:
        train_d_info = train_d(module, loader, criterion, optimizerD, device, option.gen_latent_size, option.num_labels)
        train_d_fstr = 'TrainD Loss: {:.4f}'.format(train_d_info['loss'])
    if  epoch % option.train_g_freq == 0:
        train_g_info = train_g(module, loader, criterion, optimizerG, device, option.gen_latent_size, option.num_labels)
        train_g_fstr = 'TrainG Loss: {:.4f}'.format(train_g_info['loss'])

        save_sample(os.path.join(  sample_folder, str(epoch) + '.png'), train_g_info['fake_images'])
        save_checkpoint(os.path.join(save_folder, str(epoch) + '.pth'), module, optimizerD, optimizerG, epoch)

    if  train_d_fstr != '':
        print_a_fstr += ' ' + train_d_fstr
    if  train_g_fstr != '':
        print_a_fstr += ' ' + train_g_fstr

    logger.info(print_a_fstr)
