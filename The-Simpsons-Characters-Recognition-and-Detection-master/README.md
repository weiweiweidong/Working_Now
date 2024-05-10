# The-Simpsons-Characters-Recognition-and-Detection

#### Northwestern University 

#### MSiA-432 Deep Learning

#### Spring 2020


Repository for MSiA-432 Deep Learning Course Project: The Simpsons Characters Recognition and Detection. 

Detailed information about the project can be found in the report (https://github.com/LSQI15/The-Simpsons-Characters-Recognition-and-Detection/blob/master/Final%20Report.pdf)

## Project Background
The Simpsons is a popular American animated sitcom created by Matt Groening and has been on air since 1989. As big Simpson fans, we enjoy watching the across-the-board comedy masterpiece to relax and have fun. However, because of the large number of characters, sometimes we feel like we have seen a certain character before but don't know exactly who she or he is while watching the show. This project adopted advanced convolutional neural networks VGG16 / VGG19 (Simonyan & Zisserman 2014) and Xception (Chollet, 2017) to classify images for 18 Simpson characters. In addition to image classification, this team further trained a Faster R-CNN, an object detection model, to detect and classify images with multiple characters. The deployment of this project will allow viewers to know which Simpson characters they are watching without pressing pause to check their phones.

## Dataset Description
The Simpsons Characters Data is a public data set created by Alexandre Attia. It contains images of Simpsons characters directly taken from TV show episodes and labeled by the author. The raw data contains 20,933 pictures (~45kBeach) for 48 characters. As some characters don’t have many pictures, we purposely selected top 18 characters from 48 available characters in the original data set to make sure we have enough training data (more than 350 pictures) for each character. This gave us a total of 18,992 samples. For these 18 characters, we divided pictures into 3 datasets: Training set (60%), Validation set (20%) and Test set (20%).

## Part 1: The Simpsons Characters Classification
Image classification via Convolutional neural network plays a major role in image processing, since it uses multiple filters to learn intricated feature embeddings from the original pixel under convolutional layers and adopts pooling to summary a patch of image data to understand local features. Generally speaking, a deeper network can learn substantial discriminative features to classify the images when it has an appropriate structure of convolutional and pooling layers. The team adopted several CNN structures who are the top performers in the ImageNet competition each year, including vgg16, vgg19 and Xception. 

<img src="https://github.com/LSQI15/The-Simpsons-Characters-Recognition-and-Detection/blob/master/Classification%20Result.png" width="600">

The best classification model we found was Xception with the following model parameters

    Optimizer: Adam
    Augmentation: 
        zoom_range=0.2
        rotation_range=15
        width_shift_range=0.2
        height_shift_range=0.2
        horizontal_flip=True
        vertical_flip=False
<img src="https://github.com/LSQI15/The-Simpsons-Characters-Recognition-and-Detection/blob/master/Loss%20Accuracy%20Curves.png" width="500">
<img src="https://github.com/LSQI15/The-Simpsons-Characters-Recognition-and-Detection/blob/master/Confusion%20Matrix.png" width="500">


## Part 2: The Simpsons Characters Detection

In addition to image classification, the team went a step further to implement object detection models such as the Faster R-CNN. The best performer is Faster R-CNN. It mainly consists of two parts: a region proposal network and a fast R-CNN. Region proposal network (RPN) is fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The high-quality region proposals, identified by RPN, are used by Fast R-CNN for detection. In the next step, RPN and Fast R- CNN are concatenated into a single network via shared the convolutional features, and finally generate the classification for each selected region.

In this model setting, we have the following contributions:

    1. Tuning hyperparameters related to initial part of feature representation under convolutional neural
    network, such as data augmentation, activation function, etc.
    
    2. Considering non-maximum suppression method to avoid the RPN detector overdrawing boxes
    under the same regions for one character. This implementation relies on setting a threshold
    overlapping upper bound and lower bound, which are 0.7 and 0.3 respectively.
    
    3. Producing the output images with the predicted boxes and reporting classification metric
    performance mentioned in the result section.
    
<img src="https://github.com/LSQI15/The-Simpsons-Characters-Recognition-and-Detection/blob/master/Detection%20Accuracy%20of%20Characters.png" width="600">

## References:

    Attia, A. (2017, June 12). The Simpsons characters recognition and detection using Keras (Part 1). Retrieved from https://medium.com/alex-attia-blog/the-simpsons-character-recognition-using-keras- d8e1796eae36

    Carremans, B (2018, August 17). Classify butterfly images with deep learning in Keras. Retrieved from https://towardsdatascience.com/classify-butterfly-images-with-deep-learning-in-keras-b3101fe0f98 
    
    Girshick, R (2015). Fast R-CNN. Retrieved from https://arxiv.org/abs/1504.08083

    Ren, S et al. (2015) Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. Retrieved from https://arxiv.org/abs/1506.01497
    
    He, K et al. (2015). Deep Residual Learning for Image Recognition. Retrieved from https://arxiv.org/pdf/1512.03385.pdf
    
    Krizhevsky, A et al. (2012). ImageNet Classification with Deep Convolutional Neural Networks. Retrieved from https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    
    Simonyan, K and Zisserman, A (2015).Very Deep Convolutional Networks for Large-Scale Image recognition. Retrieved from https://arxiv.org/pdf/1409.1556.pdf
    
    Szegedy, C et al. (2014). Going deeper with convolutions. Retrieved from https://arxiv.org/abs/1409.4842
