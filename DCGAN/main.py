import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.optim as optim

from models.DCGAN import Discriminator,Generator
from dataloader import get_data
from utils import weights_init
from config import params

train_dl,train_ds=get_data(params['batch_size'],params['num_workers'],params['pin_memory'])
device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

netG=Generator().to(device)
netD=Discriminator().to(device)

netG.apply(weights_init)
netD.apply(weights_init)

criterion=nn.BCELoss()
fixed_noise=torch.randn(64,params['nz'],1,1,device=device)

real_label=1.
fake_label=0.

optimizerD=optim.Adam(netD.parameters(),lr=params['lr'],betas=(params['beta1'],0.999))
optimizerG=optim.Adam(netG.parameters(),lr=params['lr'],betas=(params['beta1'],0.999))

img_list=[]
G_losses=[]
D_losses=[]
iters=0
print('start')
for epoch in range(params['epochs']):
    for i, data in enumerate(train_dl,0):
        netD.zero_grad()
        real_cpu=data[0].to(device) #batch 사이즈 가져옴
        b_size=real_cpu.size(0)
        label=torch.full((b_size,),real_label,dtype=torch.float,device=device)
        
        output=netD(real_cpu).view(-1)
        errD_real=criterion(output,label)

        errD_real.backward()
        D_x=output.mean().item()

        noise=torch.randn(b_size,params['nz'],1,1,device=device)
        fake=netG(noise)
        label.fill_(fake_label)
        output=netD(fake.detach()).view(-1)
        errD_fake=criterion(output,label)
        errD_fake.backward()
        D_G_z1=output.mean().item()
        errD=errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output=netD(fake).view(-1)
        errG=criterion(output,label)
        errG.backward()
        D_G_z2=output.mean().item()
        optimizerG.step()

        G_losses.append(errG.item())       
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == params['epochs']-1) and (i == len(train_ds)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters += 1