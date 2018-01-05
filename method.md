## Semi-Supervised face sketch synthesis based on MRF feature composition

Loss: content loss + style loss(composed gram matrix) + total variation loss  

### Content Loss
Mean square error on feature space. 
- Feature: conv1_1 layer
- Input: gray image

### Style Loss
Gram matrix based on composed sketch feature. Algorithm steps:
1. Compute input image(RGB) feature.
2. Global matching: find the top k nearest neighbour using feature from conv5_1 layer.
3. Patch matching: dense patch matching at feature space (conv3_1). 
4. Compose the target sketch feature.
5. Mean square error between predicted feature and composed features.
