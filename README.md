# mSRGAN - A Generative Adversarial Network for single image super-resolution in high content screening microscopy images.


This is the first work (to the best of my knowledge), utilizing GAN's for upscaling (4x) high content screening microscopy images and optimized for perceptual quality. Inspired by Christian et al.'s [SRGAN], a generative adversarial network, mSRGAN, is proposed for super-resolution with a perceptual loss function consisting of the weighted sum of adversarial loss, mean squared error and content loss. The objective of this implementation is to learn an end to end mapping between the low/ high-resolution images and optimize the upscaled image for quantitative metrics as well as perceptual quality.



<p align="center"><img src="https://github.com/Saurabh23/Single-Image-Super-resolution-for-high-content-screening-images-using-Deep-Learning/blob/master/thesis_scripts/prelim_results/gif22.gif" height="200" width="342" /></p>

# Motivation:

  - **High content screening image acquisition errors** : Apart from suffering the usual challenges in image acquisition (optical distortions, motion blur, noise etc.), H.C.S images are also prone to a host of domain specific challenges (Photobleaching, Cross talk, Phototoxicity, Uneven illumination, Color/Contrast errors etc.) which might further degrade the quality of the images acquired. 

  <p>
      <img src="https://github.com/Saurabh23/Single-Image-Super-resolution-for-high-content-screening-images-using-Deep-Learning/blob/master/thesis_scripts/prelim_results/micro.PNG />
  </p>
 
  - **Inefficiency of the traditional pixel wise Mean squared error (MSE)**
  - **Feature transferability issues in CNN's**




  [SRGAN]: <https://arxiv.org/abs/1609.04802>
