import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import get_data
from config import params
from models.VGG16 import VGG

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dl,valid_dl,train_ds,valid_ds=get_data()
model=VGG(params['config']).to(device)
loss_func=nn.CrossEntropyLoss(reduction='sum')
opt=optim.Adam(model.parameters(),lr=1e-3)
criterion=nn.CrossEntropyLoss()
for epoch in range(1,params['epoch']):
    for data,target in train_dl:
        model.train()
        opt.zero_grad()
        data=data.to(device)
        target=target.to(device)
        output=model(data)
        loss=criterion(output,target)
        loss.backward()
        opt.step()
    print(f'Epoch : {epoch} Loss : {loss.item():.3f}')

    for data, target in valid_dl:
        model.eval()
        with torch.no_grad():
            data=data.to(device)
            target=target.to(device)
            output=model(data)
            val_loss=criterion(output,target)
    print(f'Valid Loss : {val_loss.item():.3f}')