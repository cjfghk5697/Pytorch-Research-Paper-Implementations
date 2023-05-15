import torch.nn as nn
import torch

class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules=[]
        modules.append(nn.PixelShuffle(scale))
        self.body=nn.Sequential(*modules)
    
    def forward(self, x):
        x=self.body(x)
        return x


class make_dense(nn.Module): 
    """
    Densely Connected Convolutional Networks
    https://arxiv.org/pdf/1608.06993.pdf
    Growth_rate
    """
    def __init__(self, channels:int, growth_rate:int):
        super(make_dense,self).__init__()
        self.conv=nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x_out=nn.ReLU(self.conv(x))
        output=torch.cat((x,x_out),1)
        return output

class RDB(nn.Module):
    def __init__(self, n_layer:int, n_channels:int, growth_rate:int):
        super(RDB, self).__init__()
        modules=[]
        n_channels_re=n_channels
        self.n_layer=n_layer
        for _ in range(n_layer):
            modules.append(make_dense(n_channels_re, grwoth_rate))
            n_channels_re+=growth_rate
        self.dense_layers=nn.Sequential(*modules)
        self.conv_1x1=nn.Covn2d(n_channels_re,n_channels,kernel_size=1,padding=0,bias=False)
    def forward(self, x):
        x_out=self.conv_1x1(self.dense_layers(x))
        output=torch.cat((x_out,x),1)
        return output

class RDN(nn.Module):
    def __init__(self, n_channels:int, growth_rate:int, n_feat:int):
        super(RDN, self).__init__()
        modules=[]
        
        self.growth_rate=growth_rate
        self.n_channels=n_channels
        
        self.conv_m=nn.Conv2d(n_channels,n_feat,kernel_size=3,padding=1,bias=True)
        self.conv_0=nn.Conv2d(n_feat,n_feat,kernel_size=3,padding=1,bias=True)
        
        self.gff_1x1=nn.Conv2d(n_feat*3,n_feat kernel_size=1, padding=0, bias=True)
        self.gff_3x3=nn.Conv2d(n_feat,n_feat, kernel_size=3, padding=0, bias=True)

        self.RDB_1=RDB(n_feat, self.growth_rate)
        self.RDB_d=RDB(n_feat, self.growth_rate)
        self.RDB_D=RDB(n_feat, self.growth_rate)

        self.conv_up=nn.Conv2d(n_feat, n_feat*scale*scale,kernel_size=3,padding=1,bias=True)
        self.upsample=sub_pixel(scale)

        self.conv3=nn.Conv2d(n_feat, n_channel, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):
        f_m=self.conv_m(x)
        f_0=self.conv_0(x_0)

        f_1=self.RDB_1(f_0)
        f_d=self.RDB_d(f_1)
        f_D=self.RDB_D(f_d)
        FGF=self.gff_3x3(self.gff_1x1(torch.cat((f_1,f_d,f_D),1)))
        FDF=FGF+f_m

        us=self.conv_up(FDF)
        us=self.upsample(us)
        out=self.conv3(us)

        return out

        