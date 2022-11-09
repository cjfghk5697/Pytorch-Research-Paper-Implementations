import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data(batch_size,num_workers,image_size,pin_memory=False):
    p2d='./WGAN'

    ds=dset.STL10(
            root=p2d,
            split='train',
            download=True, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size,image_size)),
                transforms.Normalize(
                                [0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
        ]),
    )
    dl=DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    
    return dl