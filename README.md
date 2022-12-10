![204558545-dee47517-c200-4afd-b728-029b2ee7a117](https://user-images.githubusercontent.com/80466735/204776231-11904cf1-7759-4094-946b-20c234bebf1e.png)

**This repository contains proof-of-concepts for vision models demonstrated in research papers. If you wish to contribute, please contact me at cjfghk5697@gmail.com**

# Paper Implementatiom

## Contents
  * [Installation](#installation)
  * [Paper review](https://github.com/cjfghk5697/Paper_Review)
  * [Vision Implementation](#vision-implementation)
    + [GoogleNet](#googlenet)
    + [ResNet](#resnet)
    + [VGG16](#vgg16)
  * [GAN Implementation](#gan-implementation)
    + [DCGAN](#dcgan)
    + [InfoGAN](#infogan)
    + [WGAN](#wgan)
  * [Diffusion Implementation](#diffusion-implementation)
    + [DDPM](#denoising-diffusion-probabilistic-model)
    
## Installation
```
$ !git clone https://github.com/cjfghk5697/Paper_Implementation.git
$ %cd "./Pytorch-Research-Paper-Implementations/"
$ sudo pip3 install -r requirements.txt
```

## Vision Implementation
### GoogleNet

#### Abstract
We propose a deep convolutional neural network architecture codenamed "Inception", which was responsible for setting the new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014). The main hallmark of this architecture is the improved utilization of the computing resources inside the network. This was achieved by a carefully crafted design that allows for increasing the depth and width of the network while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the Hebbian principle and the intuition of multi-scale processing. One particular incarnation used in our submission for ILSVRC 2014 is called GoogLeNet, a 22 layers deep network, the quality of which is assessed in the context of classification and detection.

[Paper](https://arxiv.org/abs/1409.4842)

#### Run Example
```
$ %cd "./Pytorch-Research-Paper-Implementations/Vision Models/GoogleNet"
$ python main.py
```

### ResNet

#### Abstract
Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.
The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

[Paper](https://arxiv.org/abs/1512.03385)

#### Run Example
```
$ %cd "./Pytorch-Research-Paper-Implementations/Vision Models/ResNet"
$ python main.py
```

### VGG16

#### Abstract
In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3x3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.

[Paper](https://arxiv.org/abs/1409.1556)

#### Run Example
```
$ %cd "./Pytorch-Research-Paper-Implementations/Vision Models/VGG16"
$ python main.py
```


## GAN Implementation
### DCGAN

#### Abstract
In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.

[Paper](https://arxiv.org/abs/1511.06434)

#### Run Example
```
$ %cd "./Pytorch-Research-Paper-Implementations/GAN/DCGAN"
$ python main.py
```
### InfoGAN

#### Abstract
This paper describes InfoGAN, an information-theoretic extension to the Generative Adversarial Network that is able to learn disentangled representations in a completely unsupervised manner. InfoGAN is a generative adversarial network that also maximizes the mutual information between a small subset of the latent variables and the observation. We derive a lower bound to the mutual information objective that can be optimized efficiently, and show that our training procedure can be interpreted as a variation of the Wake-Sleep algorithm. Specifically, InfoGAN successfully disentangles writing styles from digit shapes on the MNIST dataset, pose from lighting of 3D rendered images, and background digits from the central digit on the SVHN dataset. It also discovers visual concepts that include hair styles, presence/absence of eyeglasses, and emotions on the CelebA face dataset. Experiments show that InfoGAN learns interpretable representations that are competitive with representations learned by existing fully supervised methods.

[Paper](https://arxiv.org/abs/1606.03657)

#### Run Example
```
$ %cd "./Pytorch-Research-Paper-Implementations/GAN/InfoGAN"
$ python main.py
```
### WGAN

#### Abstract
We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions.

[Paper](https://arxiv.org/abs/1701.07875)

#### Run Example
```
$ %cd "./Pytorch-Research-Paper-Implementations/GAN/WGAN"
$ python main.py
```
## Diffusion Implementation

### Denoising Diffusion Probabilistic Model

#### Abstract
We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN.

[Paper](https://arxiv.org/abs/2006.11239)

#### Run Example
```
$ %cd "./Pytorch-Research-Paper-Implementations/Diffusion/denoising diffusion probabilistic"
$ python main.py
```
