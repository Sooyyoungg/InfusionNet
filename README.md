# InfusionNet
Official Pytorch code for the paper:        
**InfusionNet: A lightweight Structure-Aware Neural Style Transfer via Feature Decomposition** (working title)            <br><br>
Notion: https://swimming-whale.notion.site/InfusionNet-A-lightweight-Structure-Aware-Neural-Style-Transfer-via-Feature-Decomposition-f991f501d9554c30861bfd9859f6e38a 

Authors:       
- 김수영 (Sooyoung Kim; rlatndud0513@snu.ac.kr) Graduate school student at Seoul National University (co-first author & corresponding author)
- 권준우 (Joonwoo Kwon; pioneers@snu.ac.kr) Graduate school student at Seoul National University (co-first author & corresponding author) <br>
- 차지욱 (Jiook Cha; cha.jiook@gmail.com) Assistant Professor at Seoul National University (co-author) <br>

## Overview
<img width="1054" alt="Model" src="https://user-images.githubusercontent.com/43199011/230831057-0a65e2f6-0649-468f-b955-96a087419bdc.png">           
<img width="1090" alt="AdaOctConv" src="https://user-images.githubusercontent.com/43199011/230831311-5d781138-3eb5-4493-b37a-3a2b75fb1bfe.png">           
**InfusonNet** is an autoencoder model consisting of content encoder, style encoder and generator. The model divides the image into two frequencies (high frequency, low frequency) using Octave Convolution. ** TOP: ** Content encoder and style encoder for extracting style information from style image. **BOTTOM:** Content encoder and generator for extracting content information from content image and infusing the style of style image to it at the generator. Adaptive Convolution is used for style transfer with Octave Convolution.


## Style Transfer
### Main Result
<p align="center"><img width="366" alt="results" src="https://user-images.githubusercontent.com/43199011/230908475-2b47763f-5d12-425a-8c69-551052bf291b.png"></p>     
- Dataset: COCO for content / WikiArt for style  <br>
Our InfusionNet learns how to preserve the content from the content image and extract the style information from the style image at each encoder. In each row, first column images are style images, second column images are content images and third column are our style transfered output images. Results images contain the important content features and also reflect the style. Our InfusionNet generates stylistic images.

### Octave Analysis
<p align="center"><img width="870" alt="Octave Analysis" src="https://user-images.githubusercontent.com/43199011/230909819-4e2e6102-c618-4546-b6cc-cd85a9667131.png"></p>       
**Left:** Our InfusionNet divides the images into two frequency ranges and learns to make their information complementary. The ratio of high frequency and low frequency means how many channels each frequency information is expressed in the channels of the feature map. We can select the ratio of high frequency and low frequency, and this is the result of the experiment. **Right:**  Our model can visualize the frequency information of the style transfered output images from the last layer in generator. High and low frequency images express well divied and complementary information of the output images.

### Loss Analysis
<p align="center"><img width="525" alt="loss_1" src="https://user-images.githubusercontent.com/43199011/230916265-29dc40cd-f73d-4248-aa7e-396b0a8897a9.png"></p>   
<p align="center"><img width="527" alt="loss_2" src="https://user-images.githubusercontent.com/43199011/230916397-064288ca-0308-4d8b-ace2-fd8b9a86a971.png"></p>
We try perceptual loss with VGG and EFDM and use EFDM for contrastive learning which is the first try. For contrastive loss, we use the nearest 1, 3, and 7 images to find the appropriate number of the compared images. We compare the learning results for each loss and decide the 'EFDM Perceptual loss + Contrastive loss with 3 images' as best result.


