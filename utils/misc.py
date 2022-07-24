import tqdm
import torch

from torchvision.utils import make_grid
from torchvision.utils import save_image

def train(module, module_id, device, latent_size, loader, criterion, optimizerD, optimizerG, train_d_freq, train_g_freq, clip_params):

    train_d_loss = []
    train_g_loss = []

    for step, mini_batch in enumerate(tqdm.tqdm(loader)):

        image , label = mini_batch
        image = image.to(device).view(image.shape[0], 1 * 28 * 28)
        label = label.to(device)

        for _ in range(train_d_freq):

            # =================
            # Fix Generator, Train Discriminator
            # =================

            module.gen.eval()
            module.dis.train()

            z = torch.randn((image.shape[0], latent_size)).to(device)

            real_target = torch.ones (image.shape[0]).to(device)
            fake_target = torch.zeros(image.shape[0]).to(device)

            real_source = image
            fake_source = module.gen(z)

            real_output = module.dis(real_source)
            fake_output = module.dis(fake_source)

            if  (
                module_id == 1
            ):
                real_loss_d = criterion(real_output, real_target)
                fake_loss_d = criterion(fake_output, fake_target)
                loss_d = (real_loss_d + fake_loss_d) / 2
            if  (
                module_id == 2
            ):
                loss_d = - torch.mean(real_output) + torch.mean(fake_output)

            optimizerD.zero_grad()
            loss_d.backward()
            optimizerD.step()

            if  module_id == 2:
                for p in module.dis.parameters():
                    p.data.clamp_(
                        -clip_params,
                        +clip_params,
                    )

            train_d_loss.append(loss_d.item())

        for _ in range(train_g_freq):

            # =================
            # Fix Discriminator, Train Generator
            # =================

            module.dis.eval()
            module.gen.train()

            z = torch.randn((image.shape[0], latent_size)).to(device)

            real_target = torch.ones(image.shape[0]).to(device)

            fake_source = module.gen(z)

            fake_output = module.dis(fake_source)

            if  (
                module_id == 1
            ):
                loss_g = criterion(fake_output, real_target)
            if  (
                module_id == 2
            ):
                loss_g = - torch.mean(fake_output)

            optimizerG.zero_grad()
            loss_g.backward()
            optimizerG.step()

            train_g_loss.append(loss_g.item())

    return {
        'train_d_loss': sum(train_d_loss) / len(train_d_loss),
        'train_g_loss': sum(train_g_loss) / len(train_g_loss),
    }

def valid(module, module_id, device, latent_size, number):

    module.gen.eval()
    module.dis.eval()

    z = torch.randn((number, latent_size)).to(device)

    fake_images = module.gen(z)

    fake_images = fake_images.view(number, 1, 28, 28)

    return {
        'fake_images': fake_images
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
