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
    return y*torch.mean(x)

for epoch in range(1,params['epoch']+1):
    for i,(data,_) in enumerate(dl):
        optim_D.zero_grad()
            
        real=data[0].to(device)
        b_size=real.size(0)
        noise=torch.randn(b_size,params['nz'],1,1,device=device)

        gen_imgs=net_G(noise).detach()
        output_data=data.view(-1)
        output_gen=gen_imgs.view(-1)
        d_loss_real=wasserstein_loss(output_data,real_label)
        d_loss_fake=wasserstein_loss(gen_imgs,fake_label)
            
        d_loss=d_loss_fake+d_loss_real
        d_loss.requires_grad_(True)
        d_loss.backward()
        optim_D.step()
        for p in net_D.parameters():
            p.data.clamp_(-params['clip_value'],params['clip_value'])

        if i % params['critic']==0:
                
            optim_G.zero_grad()
            g_img=net_G(noise)

            g_loss=wasserstein_loss(net_D(g_img),fake_label)
            g_loss.backward()
            optim_G.step()

            D_losses.append(d_loss)
            G_losses.append(g_loss)
        
    print(f'Epoch : {epoch} D Loss : {d_loss.item():.3f} G Loss : {g_loss.item():.3f}')

    if((epoch+1) == 1 or (epoch+1) == params['epochs']/2):
        with torch.no_grad():
            gen_data = net_G(fixed_noise).detach().cpu()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
        plt.savefig("Epoch_%d {}".format('MNIST') %(epoch+1))
        plt.close('all') 
    iters += 1