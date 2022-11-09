import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.optim as optim

from models.InfoGAN import Discriminator,Generator,DHead,QHead
from dataloader import get_data
from utils import weights_init,NormalNLLLoss,noise_sample
from config import params

params['num_z']=62
params['num_dis_c']=1
params['dis_c_dim']=10
params['num_con_c']=2

train_dl,train_ds=get_data(params['batch_size'],params['num_workers'],params['pin_memory'])
device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

netG=Generator().to(device)
discriminator=Discriminator().to(device)
netD=DHead().to(device)
netQ=QHead().to(device)

netG.apply(weights_init)
netD.apply(weights_init)
netQ.apply(weights_init)

criterionD=nn.BCELoss()
criterionQ_dis=nn.CrossEntropyLoss()
criterionQ_con=NormalNLLLoss()

fixed_noise=torch.randn(64,params['nz'],1,1,device=device)


optimD=optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}],lr=params['lr'],betas=(params['beta1'],0.999))
optimG=optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}],lr=params['lr'],betas=(params['beta1'],0.999))

#fixed noise
z = torch.randn(100, params['num_z'], 1, 1, device=device)
fixed_noise = z
if(params['num_dis_c'] != 0):
    idx = np.arange(params['dis_c_dim']).repeat(10)
    dis_c = torch.zeros(100, params['num_dis_c'], params['dis_c_dim'], device=device)
    for i in range(params['num_dis_c']):
        dis_c[torch.arange(0, 100), i, idx] = 1.0

    dis_c = dis_c.view(100, -1, 1, 1)

    fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)

if(params['num_con_c'] != 0):
    con_c = torch.rand(100, params['num_con_c'], 1, 1, device=device) * 2 - 1
    fixed_noise = torch.cat((fixed_noise, con_c), dim=1)

real_label=1.
fake_label=0.

img_list=[]
G_losses=[]
D_losses=[]

iters=0
print('start')
for epoch in range(params['epochs']):
    epoch_start_time = time.time()
    for i, (data,_) in enumerate(train_dl,0):
        b_size=data.size(0)
        real_data=data.to(device)

        optimD.zero_grad()
        label=torch.full((b_size,),real_label,device=device)
        output1=discriminator(real_data)
        probs_real=netD(output1).view(-1)
        loss_real=criterionD(probs_real,label)
        loss_real.backward()

        label.fill_(fake_label)
        noise,idx=noise_sample(params['num_dis_c'],params['dis_c_dim'],params['num_con_c'],params['num_z'],b_size,device)
        fake_data=netG(noise)
        output2=discriminator(fake_data.detach())
        probs_fake=netD(output2).view(-1)
        loss_fake=criterionD(probs_fake,label)

        loss_fake.backward()

        D_loss=loss_real+loss_fake
        optimD.step()
        optimG.zero_grad()

        output=discriminator(fake_data)
        label.fill_(real_label)
        probs_fake=netD(output).view(-1)
        gen_loss=criterionD(probs_fake,label)

        q_logits,q_mu,q_var=netQ(output)
        target=torch.LongTensor(idx).to(device)

        dis_loss=0
        for j in range(params['num_dis_c']):
            dis_loss+=criterionQ_dis(q_logits[:,j*10:j*10+10],target[j])

        con_loss=0
        if params['num_con_c']!=0:
            con_loss=criterionQ_con(noise[:,params['num_z']+params['num_dis_c']*params['dis_c_dim']:].view(-1,params['num_con_c']),q_mu,q_var)*0.1
        G_loss=gen_loss+dis_loss+con_loss
        G_loss.backward()
        optimG.step()
        if i != 0 and i%100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch+1, params['epochs'], i, len(train_dl), 
                    D_loss.item(), G_loss.item()))

        # Save the losses for plotting.
        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())

        iters += 1
    epoch_time = time.time() - epoch_start_time
    print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
    # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
    with torch.no_grad():
        gen_data = netG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True))

    # Generate image to check performance of generator.
    if((epoch+1) == 1 or (epoch+1) == params['epochs']/2):
        with torch.no_grad():
            gen_data = netG(fixed_noise).detach().cpu()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
        plt.savefig("Epoch_%d {}".format('MNIST') %(epoch+1))
        plt.close('all')

    # Save network weights.
    if (epoch+1) % 10 == 0:
        torch.save({
            'netG' : netG.state_dict(),
            'discriminator' : discriminator.state_dict(),
            'netD' : netD.state_dict(),
            'netQ' : netQ.state_dict(),
            'optimD' : optimD.state_dict(),
            'optimG' : optimG.state_dict(),
            'params' : params
            }, 'checkpoint/model_epoch_%d_{}'.format('MNIST') %(epoch+1))



