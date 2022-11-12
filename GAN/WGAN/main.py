import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from config import params
from dataloader import get_data
from model.model import Discriminator, Generator

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dl=get_data(params['batch_size'],params['num_workers'],params['image_size'],params['pin_memory'])

net_D=Discriminator().to(device)
net_G=Generator().to(device)

optim_D=torch.optim.RMSprop(net_D.parameters(),lr=params['lr'])
optim_G=torch.optim.RMSprop(net_G.parameters(),lr=params['lr'])

fixed_noise=torch.randn(64,params['nz'],1,1,device=device)

iters=0
G_losses=[]
D_losses=[]

for epoch in range(1,params['epoch']+1):
    for i,(data,_) in enumerate(dl):
        #discriminator train
        optim_D.zero_grad()
        data=data.to(device)
        b_size=data.size(0)

        noise=torch.randn(b_size,params['nz'],1,1,device=device) 
        
        gen_imgs=net_G(noise).detach()

        #Wasserstein-Distance 
        d_loss=torch.mean(net_D(gen_imgs))-torch.mean(net_D(data))
        d_loss.backward()
        optim_D.step()
        
        for p in net_D.parameters():
            p.data.clamp_(-params['clip_value'],params['clip_value'])
        # Discriminator train more than Generator
        if i % params['critic']==0:
            optim_G.zero_grad()
            
            g_loss=torch.mean(net_D(net_G(noise)))
            g_loss.backward()
            optim_G.step()

            D_losses.append(d_loss)
            G_losses.append(g_loss)
        
            print(f'Epoch : {epoch} D Loss : {d_loss.item():.3f} G Loss : {g_loss.item():.3f}')
    # Save image 
    if((epoch+1) == 1 or (epoch+1) == params['epoch']/2) or epoch%10==0:
        with torch.no_grad():
            gen_data = net_G(fixed_noise).detach().cpu()
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=8, padding=2, normalize=True), (1,2,0)))
        plt.savefig("Epoch_%d {}".format('Generate') %(epoch+1))
        plt.close('all') 
    iters += 1