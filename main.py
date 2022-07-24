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

from utils.misc import train, valid, save_checkpoint, load_checkpoint, save_sample

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logger.info('prepare loader')

loader = get_loader1(option)

logger.info('prepare module')

module = get_module1(option).to(device) if option.module == 1 else \
         None

logger.info('prepare envs')

optimizerD = optim.Adam(module.dis.parameters(), lr = option.lr)
optimizerG = optim.Adam(module.gen.parameters(), lr = option.lr)

criterion = nn.BCELoss()

logger.info('start training!')

for epoch in range(1, 1 + option.num_epochs):
    train_info = train(module, device, option.gen_latent_size, loader, criterion, optimizerD, optimizerG, option.train_d_freq, option.train_g_freq)
    valid_info = valid(module, device, option.gen_latent_size, option.num_labels)
    logger.info(
        '[Epoch %d] Train D Loss: %.4f, Train G Loss: %.4f' % ( epoch , train_info['train_d_loss'] , train_info['train_g_loss'] )
    )
    if  epoch % option.saved_s_freq == 0:
        save_sample(os.path.join(  sample_folder, str(epoch) + '.png'), valid_info['fake_images'])
    if  epoch % option.saved_m_freq == 0:
        save_checkpoint(os.path.join(save_folder, str(epoch) + '.pth'), module, optimizerD, optimizerG, epoch)
