# InfusionNet
Official Pytorch code for the paper:        
InfusionNet: A lightweight Structure-Aware Neural Style Transfer via Feature Decomposition (working title)

### Authors
- 김수영 (Sooyoung Kim); Graduate school student at Seoul National University
- 권준우 (Joonwoo Kwon); Graduate school student at Seoul National University

## Overview
<img width="1054" alt="Model" src="https://user-images.githubusercontent.com/43199011/230831057-0a65e2f6-0649-468f-b955-96a087419bdc.png">           
InfusonNet is an autoencoder model consisting of content encoder, style encoder and generator. The model divides the image into two frequencies (high frequency, low frequency) using Octave Convolution. TOP: Content encoder and style encoder for extracting style information from style image. BOTTOM: Content encoder and generator for extracting content information from content image and infusing the style of style image to it at the generator.

<img width="1090" alt="AdaOctConv" src="https://user-images.githubusercontent.com/43199011/230831311-5d781138-3eb5-4493-b37a-3a2b75fb1bfe.png">          
Adaptive Convolution is used for style transfer with Octave Convolution.

## Style Transfer


## Style Mixing
