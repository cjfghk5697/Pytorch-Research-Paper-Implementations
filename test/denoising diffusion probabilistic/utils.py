from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from config import params
from torchvision import transforms 
import numpy as np

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

class DenoiseDiffusion:
  def __init__(self,eps_model:nn.Module,n_steps:int,device:torch.device):
    super().__init__()
    self.eps_model=eps_model
    self.beta=torch.linspace(0.0001,0.02,n_steps).to(device)
    self.alpha=1-self.beta
    self.alpha_bar=torch.cumprod(self.alpha,dim=0)
    self.n_steps=n_steps
    self.sigma2=self.beta
    self.device=device
  
  def q_xt_x0(self,x_0:torch.Tensor,t:torch.Tensor):
    mean=gather(self.alpha_bar,t)**0.5*x_0
    var=1-gather(self.alpha_bar,t)
    return mean,var

  def q_sample(self,x0:torch.Tensor,t:torch.Tensor,eps:Optional[torch.Tensor]=None):
    if eps is None:
      eps=torch.rand_like(x0)
    mean,var=self.q_xt_x0(x0,t)
    return mean+(var**0.5)*eps
  
  def p_sample(self,xt:torch.Tensor,t:torch.Tensor):
    eps_theta=self.eps_model(xt,t)
    alpha_bar=gather(self.alpha_bar,t)
    alpha=gather(self.alpha,t)
    eps_coef=(1-alpha)/(1-alpha_bar)**0.5
    mean=(1/(alpha**0.5))*(xt-eps_coef*eps_theta)
    var=gather(self.sigma2,t)
    eps=torch.randn(xt.shape,device=xt.device)
    return mean+(var**0.5)*eps
  
  def loss(self,x0:torch.Tensor,noise:Optional[torch.Tensor]=None):
    batch_size=x0.shape[0]
    t=torch.randint(0,self.n_stpes,(batch_size,),device=x0.device,dtype=torch.long)
    if noise is None:
      noise=torch.randn_lise(x0)
    xt=self.q_sample(x0,t,eps=noise)
    eps_theta=self.eps_model(xt,t)
    return F.mse_loss(noise,eps_theta)

@torch.no_grad()
def sample_plot_image(t):
    # Sample noise
    img_size = params['image_size']
    img = torch.randn((1, 1, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=self.device, dtype=torch.long)
        img = self.p_sample(img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, i/stepsize+1)
            show_tensor_image(img.detach().cpu())
    plt.savefig(f"{t} epochs")
    plt.show()       


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ]) 
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image), cmap=plt.cm.gray)