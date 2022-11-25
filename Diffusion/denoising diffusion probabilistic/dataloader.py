import torchvision.datasets as dset
import torchvision.transforms as transforms
from config import params

from torch.utils.data import DataLoader

def get_data():
    p2d='./diffusion'

    train_ds=dset.StanfordCars(
        root=p2d,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((params['image_size'],params['image_size'])),
            transforms.Lambda(lambda t: (t*2)-1)
        ])
    )


    train_dl=DataLoader(
        train_ds,
        batch_size=params['batch_size'],
        pin_memory=params['pin_memory'],
        num_workers=params['num_workers']
    )


    return train_dl