import numpy as np
from matplotlib import pyplot as plt
def show(img,y=None):
    npimg=img.numpy()
    npimg_tr=np.transpose(npimg,(1,2,0))
    plt.imshow(npimg_tr)
    if y is not None:
        plt.title('lables : '+str(y))

