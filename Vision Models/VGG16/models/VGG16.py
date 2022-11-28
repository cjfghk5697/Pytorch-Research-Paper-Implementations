import torch.nn as nn

class Conv(nn.Module):
    """
    Convoulution Conv->ReLU Layer.
    VGG 16 padding is same.
    """
    def __init__(self,in_channels,out_channels):
        super(Conv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )

    def forward(self,x):
        return self.conv(x)

class VGG(nn.Module):
    """
    To Create VGG 
    """

    def __init__(self,config,num_classes=1000):
        super(VGG,self).__init__()
        self.config=config
        self.FC=nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,num_classes)
            )
        
        self.layer=self.make_layer()
        
    def make_layer(self):
        in_planes=3
        layers=[]
        for component in self.config:
            if component=='M':
                layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
            else:
                layers.append(Conv(in_planes,component))
                in_planes=component
        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.layer(x)
        x=x.view(-1,512*7*7)
        x=self.FC(x)
        return x

