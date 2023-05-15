from typing import Optional, List
import numpy as np
import torch

from labml_nn.diffusion.stable_

def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

class DDIM:
    def __init__(self, eps_model: nn.Module, n_steps: int,device: torch.device):
        super().__init__()
        self.eps_model=eps_model
        self.beta=torch.linspace(1e-4, 2e-3, n_steps).to(device)
        self.alpha=1-self.beta
        self.alpha_bar=torch.cumprod(self.alpha, dim=0)
        self.n_steps=eps_model.n_steps
        self.sigma2=self.beta
        """
        ODE model need about model data like timestep.
        """
        c=self.n_steps//n_steps
        self.time_steps=np.asarray(list(range(0,self.n_steps,c)))+1
        alpha_bar=self.model.alpha_bar
        self.ode_alpha=alpha_bar[self.time_stpes].clone().to(torch.float32)
        self.ode_alpha_sqrt=torch.sqrt(self.ode_alpha)
        self.ode_alpha_prev=torch.cat([alpha_bar[0:1]])
        self.ode_sigma=()


    def q_xt_x0(self, x_0:torch.Tensor, t:torch.Tensor):
        mean=gather(self.alpha_bar,t)**0.5*x_0
        var=1-gather(self.alpha_bar, t)
        return mean, var
    
    def q_sample(self, x0:torch.Tensor, t:torch.Tensor):
        if eps in None:
            eps=torch.rand_like(x0)
        mean,var=self.q_xt_x0(x0,t)

        return mean+(var**0.5)*eps

    def p_sample(self, xt:torch.Tensor, t:torch.Tensor):
        """
        algorithm 1
        """
        eps_theta=self.eps_model(xt,t)
        alpha_bar=gather(self.alpha_bar,t)
        alpha=gather(self.alpha,t)
        eps_coef=(1-alpha)/(1-alpha_bar)**.5
        mean=1/(alpha**.5)*(xt-eps_coef*eps_theta)
        
        var=gather(self.sigma2,t)
        eps=torch.randn(xt.shape, deivce=xt.device)
        
        return mean+(var**.5)*eps
    
    def dx_t(self, xt:torch.Tensor, t:torch.Tensor):
        sample=self.p_sample(xt,t)

    
    def loss(self, x0:torch.Tensor):
        batch_size=x0.shape[0]
        t=torch.randint(0,self.n_steps,(batch_size,),device=x0.device,dtype=torch.long)
        noise=torch.randn_like(x0)
        xt=self.q_sample(x0,t,eps=noise)
        eps_theta=self.eps_model(xt,t)
        return F.l1_loss(noise,eps_theta)
