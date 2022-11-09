import torch
import numpy as np
from config import params
from dataloader import get_data
from model.model import Discriminator, Generator
import torchvision.utils as vutils

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dl=get_data(params['batch_size'],params['num_workers'],params['image_size'],params['pin_memory'])

net_D=Discriminator()
net_G=Generator()

optim_G=torch.optim.RMSprop(net_G.parameters(),lr=params['lr'])
optim_D=torch.optim.RMSprop(net_G.parameters(),lr=params['lr'])
fixed_noise=torch.randn(64,params['nz'],1,1,device=device)

real_label=-1.
fake_label=1.

iters=0
img_list=[]
G_losses=[]
D_losses=[]

def wasserstein_loss(x,y):
    return torch.mean(x*y)

for epoch in range(1,params['epoch']+1):
    for cirtic in range(params['critic']):
        for i,(data,_) in enumerate(dl):
            optim_D.zero_grad()
            
            real=data[0].to(device)
            b_size=real.size(0)
            r_label=torch.full((b_size,),real_label,dtype=torch.float,device=device)
            noise=torch.randn(b_size,params['nz'],1,1,device=device)

            gen_imgs=net_G(noise).detach()
            f_label=torch.full((b_size,),real_label,dtype=torch.float,device=device)
            
            d_loss_real=wasserstein_loss(data,r_label)
            d_loss_fake=wasserstein_loss(gen_imgs,f_label)
            d_loss=0.5*np.add(d_loss_fake,d_loss_real)
            
            d_loss.backward()
            optim_D.step()
            for p in net_D.parameters():
                p.data.clamp_(-params['clip_value'],params['clip_value'])


            
    optim_G.zero_grad()
    g_img=net_G(noise)

    g_loss=wasserstein_loss(net_D(g_img),f_label)
    g_loss.backward()
    optim_G.step()

    D_losses.append(d_loss)
    G_losses.append(g_loss)
    
    print(f'Epoch : {epoch} D Loss : {d_loss} G Loss : {g_loss}')

    if (iters % 500 == 0) or ((epoch == params['epochs']-1)):
        with torch.no_grad():
            fake = net_G(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    iters += 1