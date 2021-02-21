# End-to-end dog breads classification

In this notebook i'm going to build a multi-class image classifier using TensorFlow.

## 1. Problem
Identifying the bread of a dog given an image of a dog.
When we've seen a dog and don't know its bread, we can take a picture of that and understanding its bread.

## 2. Data
The data we're going to using is from Kaggle Dog Bread Identification competition.
https://www.kaggle.com/c/dog-breed-identification/overview

## 3. Evaluation
The evaluation is a file with prediction probebilities for each dog bread of each image test.
https://www.kaggle.com/c/dog-breed-identification/overview/evaluation

## 4. Features

Some information about the data:

* We're dealing with images (unstructured data), so it's probably best to use deeplearning/transfer learning.
* There are 120 breads of dogs (means there are 120 classes).
* There are around 10,000+ images in training set (with labels).
* There are around 10,000+ images in testing set (without labels).

## 5. Model

The model I used is [MobileNet_v2](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4)