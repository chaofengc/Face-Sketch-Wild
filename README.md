# End to End Face Sketch Synthesis

## Related works

### Traditional Method

1. *Face Sketch Synthesis and Recognition. (Tang and Wang, ICCV2003)* **The very first paper**  
Key ideas:
  - A new face can be reconstructed from training samples by PCA.
  - The transformation between photo and sketch can be approximated as a linear process.

1. *A Nonlinear Approach for Face Sketch Synthesis and Recognition (CVPR2005)*  
Key ideas: extract images into patches, assume local linearity rather than global linearity

#### Other works based on local patches
- :star: Face Photo-Sketch Synthesis and Recognition.    (PAMI 2009)  
- Lighting and pose robust face sketch synthesis. (ECCV2010)  
- Markov Weight Fields for Face Sketch Synthesis. (CVPR2012)  
- Real-Time Exemplar-Based Face Sketch Synthesis. (ECCV2014)  

### Deep Learning Method.

### FCN based Approach
- End-to-End Photo-Sketch Generation via Fully Convolutional Representation Learning.
- :star: Content-Adaptive Sketch Portrait Generation by Decompositional Representation Learning.

### GAN based Approach

- Recursive Cross-Domain Face/Sketch Generation from Limited Facial Parts. (ICML Workshop)
- :star: High-Quality Facial Photo-Sketch Synthesis Using Multi-Adversarial Networks. (Arxiv)

- :star: Image-to-Image Translation with Conditional Adversarial Networks. (CVPR2017) [Github](https://github.com/phillipi/pix2pix)
- :star: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. (ICCV2017) [Github](https://github.com/junyanz/CycleGAN)

### Style transfer Approach
- Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks. [Github](https://github.com/chuanli11/MGANs)
- Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis. [Github](https://github.com/chuanli11/CNNMRF)
