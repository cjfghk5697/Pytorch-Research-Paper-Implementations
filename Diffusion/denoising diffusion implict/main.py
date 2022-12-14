from models.model import Unet
from dataloader import get_data
from utils import get_loss,sample_plot_image
from config import params
import torchvision.transforms as transforms
import torch.optim as optim
import torch

model=Unet(
    dim=params['image_size'],
    channels=1,
    dim_mults=(1, 2, 4,))
device="cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer=optim.Adam(model.parameters(), lr=params['lr'])
epochs=params['epochs']

transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((params['image_size'],params['image_size'])),
        transforms.Lambda(lambda t: (t * 2) - 1)
])
dataloader=get_data(transforms)
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        t=torch.randint(0,params['T'],(batch[0].shape[0],),device=device).long()
        loss=get_loss(model,batch[0],t)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 and step == 0:
          print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
          sample_plot_image(model,epoch)

