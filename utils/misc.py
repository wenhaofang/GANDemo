import tqdm
import torch

from torchvision.utils import make_grid
from torchvision.utils import save_image

def train_d(module, loader, criterion, optimizerD, device, latent_size, number):
    '''
    Fix Generator, Train Discriminator
    '''
    module.gen.eval()
    module.dis.train()
    epoch_loss = 0.0

    for mini_batch in tqdm.tqdm(loader):
        image , label = mini_batch
        image = image.to(device).view(image.shape[0], 1 * 28 * 28)
        label = label.to(device)

        real_target = torch.ones (image.shape[0]).to(device)
        fake_target = torch.zeros(image.shape[0]).to(device)

        z = torch.randn((image.shape[0], latent_size)).to(device)

        real_source = image
        fake_source = module.gen(z)

        real_output = module.dis(real_source)
        fake_output = module.dis(fake_source)

        real_loss_d = criterion(real_output, real_target)
        fake_loss_d = criterion(fake_output, fake_target)

        loss_d = (real_loss_d + fake_loss_d) / 2

        optimizerD.zero_grad()
        loss_d.backward()
        optimizerD.step()

        epoch_loss += loss_d.item()

    return {
        'loss': epoch_loss / len(loader)
    }

def train_g(module, loader, criterion, optimizerG, device, latent_size, number):
    '''
    Fix Discriminator, Train Generator
    '''
    module.dis.eval()
    module.gen.train()
    epoch_loss = 0.0
    true_images = []
    fake_images = []

    for mini_batch in tqdm.tqdm(loader):
        image , label = mini_batch
        image = image.to(device).view(image.shape[0], 1 * 28 * 28)
        label = label.to(device)

        real_target = torch.ones (image.shape[0]).to(device)

        z = torch.randn((image.shape[0], latent_size)).to(device)

        fake_source = module.gen(z)

        fake_output = module.dis(fake_source)

        loss_g = criterion(fake_output, real_target)

        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()

        epoch_loss += loss_g.item()
        true_images.append(image)
        fake_images.append(fake_source)

    true_images = torch.cat(true_images, dim = 0)
    fake_images = torch.cat(fake_images, dim = 0)
    true_images = true_images.view(true_images.shape[0], 1, 28, 28)
    fake_images = fake_images.view(fake_images.shape[0], 1, 28, 28)
    true_images = true_images[:number]
    fake_images = fake_images[:number]

    return {
        'loss': epoch_loss / len(loader),
        'true_images': true_images,
        'fake_images': fake_images,
    }

def save_checkpoint(save_path, module, optimD, optimG, epoch):
    checkpoint = {
        'module': module.state_dict(),
        'optimD': optimD.state_dict(),
        'optimG': optimG.state_dict(),
        'epoch' : epoch,
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(load_path, module, optimD, optimG):
    checkpoint =torch.load(load_path)
    module.load_state_dict(checkpoint['module'])
    optimD.load_state_dict(checkpoint['optimD'])
    optimG.load_state_dict(checkpoint['optimG'])
    return checkpoint['epoch']

def save_sample(save_path, samples):
    save_image (make_grid (samples.cpu(), nrow = 5, normalize = True).detach(), save_path)
