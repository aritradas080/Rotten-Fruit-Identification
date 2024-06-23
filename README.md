# Rotten Fruit Identification with Vision Transformer (both Build-from-Scratch model and Pretrained model)

# Introduction to ViT: 

Vision Transformer (ViT) uses transformer architecture for computer vision applications, particularly image categorization. In a variety of image identification tasks, Vision Transformers have demonstrated outstanding performance, frequently outperforming conventional convolutional neural networks (CNNs).

Originally developed for natural language processing (NLP), transformers use self-attention methods to manage sequential data. This enables transformers to dynamically determine the relative relevance of various input data components.

# To address Computer vision tasks - 
First, images are separated into smaller, fixed-size patches, which are then used to adapt transformers for vision tasks. After that, each patch is compressed and linearly integrated into a vector. These vectors are sent into the transformer as a sequence, much like word embeddings in natural language processing.

ViTs have Self-attention mechanism which allows the ViT models to understand the relationships between different parts of an image by assigning priority scores to patches and focusing on the most relevant information. This helps the model make better sense of the image and perform various tasks related to computer vision. 

# Number of patches:

If an image has X * X pixels and we want to create a patch of y * y pixels; then the number of patches will be (X/Y) * (X/Y). For any RGB image, the patch will have a size of (X/Y)*(X/Y)*3 due to three color channels. 
# Structure of a Vision Transformer and the step-by-step processing of an input image within the vision transformer (ViT)
a. Flattening the patches into one dimensional vector:
The patches are flattened into one dimensional vector.

b. Linear projection :
Each 1D vector is transformed into a lower dimensional vector by linear projection, maintaining key characteristics and relationships in the process. Linear projection consists of two main steps: One is weight matrix multiplication and bias addition. The weights and biases are learnt during the training process. These two steps combindly produce a transformed vector of lower dimensionality. The benefits of reducing dimensionality are –
1.	Faster computation and lower memory consumption
2.	Most significant features become extractable.
3.	Reduces noises and irrelevant information.

All the vectors are passed through a linear layer to create Patch Embeddings.

c. Positional Encoding:
Next, Positional Encodings are added to the Patch Embeddings in order to preserve spatial information, as transformers do not grasp the order or position of patches by default.

d. Transformer Encoder:
Transformer encoders are layered in order to pass the encoded patches. Feed-forward neural networks and multi-head self-attention processes make up each encoder layer. Self-attention layers try to extract dependencies and information from other patches. Additionally, the model is able to comprehend the global context thanks to self-attention layers. Feed forward networks come after the layer of self-attention. The complex non-linear interactions between the patches are better captured by them. 





## Structure of ViT

![App Screenshot](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png)


## About this project
# Dataset description
In this project, a dataset with images of fresh and rotten fruits has been taken. There are 6 classes and there are almost 13,600+ images. The classes are -
a) freshapples,
b) freshbanana,
c) freshoranges,
d) rottenapples,
e) rottenbanana,
f) rottenoranges

# Hyperparameters (For both, Build-from-Scratch model and Pretrained model),

Batch size = 32

Number of epochs = 5

Optimizer = Adam 

Image size = 224

Initial Learning Rate = 1e-3

Decay rate = 0.2

# Data Augmentation techniques (For both, Build-from-Scratch model and Pretrained model),

1. Gaussian Blur
2. Adding Gaussian Noise
3. Random Flip (Horizontal and Vertical)
4. Random Image Rotation
5. Random Brightness
6. Contrast Adjustment

# Results

## For Build-from-Scratch ViT model

1. train_loss: 1.7819 
2. train_accuracy: 0.2113 
3. test_loss: 1.7822 
4. test_accuracy: 0.2210

## For Pretrained ViT model
1. train_loss: 0.0290 
2. train_accuracy: 0.9949 
3. test_loss: 0.1949 
4. test_accuracy : 0.9357

# References
1. https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png

2. Link of the dataset : https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

3. https://arxiv.org/abs/2010.11929 ("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale")



