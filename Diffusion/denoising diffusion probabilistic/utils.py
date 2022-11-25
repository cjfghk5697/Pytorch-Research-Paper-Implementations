import torch.nn.functional as F
import torch
def linear_beta_schedule(timesteps,start=0.0001,end=0.02):
    return torch.linspace(start,end,timesteps)

def get_index_from_list(vals,t,x_shape):
    batch_size=t.shape[0]
    out=vals.gather(-1,t.cpu())
    return out.reshape(batch_size,*((1,)*(len(x_shape)-1))).to(t.device)

def forward_diffusion_sample(x_0,t,device='cpu'):
    noise=torch.randn_like(x_0)
    sqrt_alphas_cumprod_t=get_index_from_list(sqrt_alphas_cumprod,t,x_0.shape)
    sqrt_one_minus_alphas_cumprod_t=get_index_from_list(
        sqrt_one_minus_alphas_cumprod,t,x_0.shape
    )
    return sqrt_alphas_cumprod_t.to(device)*x_0.to(device)\
        + sqrt_one_minus_alphas_cumprod_t.to(device)*noise.to(device), noise.to(device)

T=300
betas=linear_beta_schedule(timesteps=T)

alphas=1.-betas
alphas_cumprod=torch.cumprod(alphas,axis=0)
alphas_cumprod_prev=F.pad(alphas_cumprod[:-1],(1,0),value=1.0)
sqrt_recip_alphas=torch.sqrt(1.0/alphas)
sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod=torch.sqrt(1.-alphas_cumprod)
posterior_variance=betas*(1.-alphas_cumprod_prev)/(1.-alphas_cumprod)