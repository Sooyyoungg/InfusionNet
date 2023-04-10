# InfusionNet
Official Pytorch code for the paper:        
InfusionNet: A lightweight Structure-Aware Neural Style Transfer via Feature Decomposition (working title)

Authors:       
- 김수영 (Sooyoung Kim; rlatndud0513@snu.ac.kr) Graduate school student at Seoul National University
- 권준우 (Joonwoo Kwon; pioneers@snu.ac.kr) Graduate school student at Seoul National University

## Overview
<img width="1054" alt="Model" src="https://user-images.githubusercontent.com/43199011/230831057-0a65e2f6-0649-468f-b955-96a087419bdc.png">           
<img width="1090" alt="AdaOctConv" src="https://user-images.githubusercontent.com/43199011/230831311-5d781138-3eb5-4493-b37a-3a2b75fb1bfe.png">           
__InfusonNet__ is an autoencoder model consisting of content encoder, style encoder and generator. The model divides the image into two frequencies (high frequency, low frequency) using Octave Convolution. __TOP:__ Content encoder and style encoder for extracting style information from style image. __BOTTOM:__ Content encoder and generator for extracting content information from content image and infusing the style of style image to it at the generator. Adaptive Convolution is used for style transfer with Octave Convolution.


## Style Transfer
### Main Result
<p align="center"><img width="366" alt="results" src="https://user-images.githubusercontent.com/43199011/230908475-2b47763f-5d12-425a-8c69-551052bf291b.png"></p>     


### Octave Analysis
<p align="center"><img width="870" alt="Octave Analysis" src="https://user-images.githubusercontent.com/43199011/230909819-4e2e6102-c618-4546-b6cc-cd85a9667131.png"></p>       


## Style Mixing
