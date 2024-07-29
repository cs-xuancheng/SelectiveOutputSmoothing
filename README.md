# Selective Output Smoothing Regularization: Regularize Neural Networks by Softening Output Distributions

This is a pytorch implementation of SOSR in ImageNet.

## Main Requirements

- torch == 1.0.1
- torch == 0.2.0
- Python 3

## Usage
To train the ResNet-50 with P=0.99, $\beta = 0.5$, please specify the location of ImageNet first:

`python imagenet_sosr.py --depth 50 --data your_data_location --threshold 0.99 --beta 0.5 --gpu-id 0,1` 