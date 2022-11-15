import torchvision.datasets as dset
import torchvision.transforms as transforms
from config import params
from torch.utils.data import DataLoader

def get_data():
    p2d='./Vision Models/ResNet/dataset'

    ds=dset.STL10(
        root=p2d,
        split='train',
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((params['img_size'],params['img_size'])),
            transforms.Normalize([0.5],[0.5]),
        ])
    )
    dl=DataLoader(
        ds,
        batch_size=params['batch'],
        pin_memory=params['pin'],
        num_workers=params['num_workers']
    )

    return dl,ds