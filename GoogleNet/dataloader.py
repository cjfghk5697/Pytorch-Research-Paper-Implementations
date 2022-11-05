import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
def get_data(batch_size,num_workers,pin_memory=False):
    path2data='/workspaces/codespace/GoogleNet'

    train_ds=datasets.STL10(path2data,split='train',
                        download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(224),
                            transforms.Normalize(        
                                [0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
                        ])
                        )
    val_ds=datasets.STL10(path2data,split='test',
                        download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(224),
                            transforms.Normalize(        
                                [0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
                        ])
                        )

    
    train_dl=DataLoader(train_ds,batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory
                        )
    valid_dl=DataLoader(val_ds,batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory
                        )
    return train_dl,valid_dl,train_ds,val_ds
