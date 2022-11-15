import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data(batch_size,num_workers,image_size,pin_memory=False):
    p2d='./WGAN'

    ds=dset.MNIST(
            root=p2d,
            train=True,
            download=True, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])]),
    )
    dl=DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    
    return dl