import torchvision.datasets as dset
import torchvision.transforms as transforms
from config import params

from torch.utils.data import DataLoader

def get_data(batch_size,num_workers,pin_memory=False):
    p2d='./InfoGAN'

    train_ds=dset.STL10(
        root=p2d,
        split='train',
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((params['image_size'],params['image_size'])),
            transforms.Normalize(
                            [0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
        ])
    )


    train_dl=DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers
    )


    return train_dl,train_ds