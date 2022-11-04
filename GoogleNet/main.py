import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.optim as optim
import copy

from torch.optim.lr_scheduler import StepLR
from models.stl_model import GoogLeNet
from matplotlib import pyplot as plt
from config import params
from visualization import show
from dataloader import get_data 

np.random.seed(0)
torch.manual_seed(0)

train_dl,valid_dl,train_ds,valid_ds=get_data(params['batch_size'],params['pin_memory'])

grid_size=4
rnd_inds=np.random.randint(0,len(train_ds),grid_size)
print("image indicies : ",rnd_inds)

x_grid=[train_ds[i][0] for i in rnd_inds]
y_grid=[train_ds[i][1] for i in rnd_inds]

x_grid=vutils.make_grid(x_grid,nrow=4,padding=2)
print(x_grid.shape)

plt.figure(figsize=(10,10))
show(x_grid,y_grid)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=GoogLeNet(aux_logits=True,num_classes=10,init_weights=True).to(device)

loss_func=nn.CrossEntropyLoss(reduction='sum')
opt=optim.Adam(model.parameters(),lr=1e-3)
lr_scheduler=StepLR(opt,step_size=30,gamma=0.1)

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def metric_batch(output,target):
    pred=output.argmax(dim=1,keepdim=True)
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func,outputs,target,opt=None):
    if np.shape(outputs)[0]==3:
        output,aux1,aux2=outputs
        output_loss=loss_func(output,target)
        aux1_loss=loss_func(aux1,target)
        aux2_loss=loss_func(aux2,target)

        loss=output_loss+0.3*(aux1_loss+aux2_loss)
        metric_b=metric_batch(output,target)
    else:
        loss=loss_func(outputs,target)
        metric_b=metric_batch(outputs,target)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    return loss.item(),metric_b

def loss_epoch(model, loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data=len(dataset_dl.dataset)

    for xb,yb in dataset_dl:
        xb=xb.to(device)
        yb=yb.to(device)
        output=model(xb)

        loss_b,metric_b=loss_batch(loss_func,output,yb,opt)

        running_loss+=loss_b
        if metric_b is not None:
            running_metric+=metric_b
        if sanity_check is True:
            break

    loss=running_loss/len_data
    metric=running_metric/len_data

    return loss,metric

def train_val(model,params):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    opt=params['optimizer']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    sanity_check=params['sanity_check']
    lr_scheduler=params['lr_scheduler']
    path2weights=params['path2weights']

    loss_history={'train':[],'val':[]}
    metric_history={'train':[],'val':[]}

    best_model_wts=copy.deepcopy(model.state_dict())

    best_loss=float('inf')

    start_time=time.time()
    for epoch in range(params['num_epochs']):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))

        model.train()
        train_loss,train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,opt)  
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss,val_metric=loss_epoch(model,loss_func,val_dl,sanity_check)
        
        if val_loss<best_loss:
            best_loss=val_loss
            best_model_wts=copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(),path2weights)
            print('Copy')
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)
        lr_scheduler.step()

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
        print('-'*10)
    model.load_state_dict(best_model_wts)

    return model, loss_history, metric_history

params_train = {
    'num_epochs':100,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_dl':train_dl,
    'val_dl':valid_dl,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/weights.pt',
}

model,loss_hist,metric_hsit=train_val(model,params_train)