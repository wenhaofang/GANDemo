from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

def get_loader(option):
    dataset = datasets.MNIST(root = option.path, train = True, download = True, transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5], std = [0.5]),
    ]))
    dataloader = DataLoader(dataset = dataset, batch_size = option.batch_size, shuffle = True)

    return dataloader

if  __name__ == '__main__':

    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    loader = get_loader(option)

    for mini_batch in loader:
        image = mini_batch[0]
        label = mini_batch[1]
        print(image.shape) # (batch_size, 1, 28, 28)
        print(label.shape) # (batch_size)
        break
