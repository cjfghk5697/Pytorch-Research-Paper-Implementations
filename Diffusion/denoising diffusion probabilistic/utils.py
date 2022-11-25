from config import params
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

device="cuda" if torch.cuda.is_available() else "cpu"

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

T=params['T']
betas=linear_beta_schedule(timesteps=T)

alphas=1.-betas
alphas_cumprod=torch.cumprod(alphas,axis=0)
alphas_cumprod_prev=F.pad(alphas_cumprod[:-1],(1,0),value=1.0)
sqrt_recip_alphas=torch.sqrt(1.0/alphas)
sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod=torch.sqrt(1.-alphas_cumprod)
posterior_variance=betas*(1.-alphas_cumprod_prev)/(1.-alphas_cumprod)

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img_size = params['image_size']
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, i/stepsize+1)
            show_tensor_image(img.detach().cpu())
    plt.show()            