import torch.nn as nn
from config import params

class G_conv(nn.Module):
    def __init__(self,in_channels,out_channels,*args):
        super(G_conv,self).__init__()
        self.conv=nn.Sequential(
            nn.ConvTranspose2d( in_channels,out_channels,*args, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
    def forward(self,x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.main=nn.Sequential(
            G_conv(params['nz'],params['ngf']*8,4,1,0),
            G_conv(params['ngf']*8,params['ngf']*4,4,2,1),
            G_conv(params['ngf']*4,params['ngf']*2,4,2,1),
            G_conv(params['ngf']*2,params['ngf'],4,2,1),

            nn.ConvTranspose2d(params['ngf'],params['nc'],4,2,1,bias=False),
            nn.Tanh()
        )
    
    def forward(self,x):
        return self.main(x)

class D_conv(nn.Module):
    def __init__(self,in_channels,out_channels,*args):
        super(D_conv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self,x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main=nn.Sequential(
            nn.Conv2d(params['nc'],params['ndf'],4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            D_conv(params['ndf'],params['ndf']*2,4,2,1),
            D_conv(params['ndf']*2,params['ndf']*4,4,2,1),
            D_conv(params['ndf']*4,params['ndf']*8,4,2,1),
            nn.Conv2d(params['ndf']*8,1,4,1,0,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.main(x)