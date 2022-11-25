from models.model import SimpleUnet
from dataloader import get_data
from utils import params,get_loss,sample_plot_image
import torch.optim as optim
import torch

model=SimpleUnet()
device="cuda" if torch.cuda.is_availabe() else "cpu"
model.to(device)
optimizer=optim.Adam(model.parameters(), lr=params['lr'])
epochs=params['epochs']
dataloader=get_data()
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        t=torch.randint(0,params['T'],(params['batch_size'],),device=device).long()
        loss=get_loss(model,batch[0],t)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            sample_plot_image()

