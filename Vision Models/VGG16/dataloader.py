from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from config import params
def get_data():
    p2d='./VGG16'
    
    train_ds=datasets.STL10(
        p2d,
        split='train',
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        )
    valid_ds=datasets.STL10(
        p2d,
        split='test',
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        )

    train_dl=DataLoader(
        train_ds,
        batch_size=params['batch_size'],
        num_workers=params['num_workers'],
        pin_memory=params['pin_memory']
        )

    valid_dl=DataLoader(
        valid_ds,
        batch_size=params['batch_size'],
        num_workers=params['num_workers'],
        pin_memory=params['pin_memory']
        )
    
    return train_dl,valid_dl,len(train_dl),len(valid_dl)