import torch
import torch.nn as nn

from config import params
from dataloader import get_data
from models.ResNet import resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


dl,ds=get_data()
model=resnet50().to(device)

optim=torch.optim.AdamW(model.parameters(),lr=params['lr'])
cirterion=nn.CrossEntropyLoss()
losses=[]


for epoch in range(1,params['epoch']+1):
    for data,target in dl:
        #discriminator train
        model.train()
        optim.zero_grad()
        data=data.to(device)
        target=target.to(device)
        output=model(data)
        loss=cirterion(output,target)
        loss.backward()
        optim.step()
        print(f'Epoch : {epoch} Loss : {loss.item():.3f}')
