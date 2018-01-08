# mSRGAN - A Generative Adversarial Network for single image super-resolution in high content screening microscopy images.


This is the first work (to the best of my knowledge), utilizing GAN's for upscaling (4x) high content screening microscopy images and optimized for perceptual quality. Inspired by Christian et al.'s [SRGAN], a generative adversarial network, mSRGAN, is proposed for super-resolution with a perceptual loss function consisting of the weighted sum of adversarial loss, mean squared error and content loss. The objective of this implementation is to learn an end to end mapping between the low/ high-resolution images and optimize the upscaled image for quantitative metrics as well as perceptual quality.



<p align="center"><img src="https://github.com/Saurabh23/Single-Image-Super-resolution-for-high-content-screening-images-using-Deep-Learning/blob/master/thesis_scripts/prelim_results/gif22.gif" height="300" width="342" /></p>

# Motivation:

  - **High content screening image acquisition errors** : Apart from suffering the usual challenges in image acquisition (optical distortions, motion blur, noise etc.), H.C.S images are also prone to a host of domain specific challenges (Photobleaching, Cross talk, Phototoxicity, Uneven illumination, Color/Contrast errors etc.) which might further degrade the quality of the images acquired. 

 
  - **Inefficiency of the traditional pixel wise Mean squared error (MSE)**: M.S.E has a lot of flaws for generating images, and images produced by MSE do not correlate well with the image quality perceived by a human observer. M.S.E overly penalizes larger errors, while is more forgiving to the small errors ignoring the underlying structure of the image. M.S.E tends to have more local minima which make it challenging to reach convergence towards a better local minimum. Consequently, the most common metric to quantitatively measure the image quality, p.s.n.r, corresponds poorly to a human's perception of an image quality.
  
  - **Feature transferability between distant source and target domains in CNN's**: Transferability of features from convnets is inversely proportional to the distance between source and target tasks, ([Bengio et al.], [Azizpour et al.]). Hence, directly using [SRGAN] (which minimises feature representations from a VGG19 trained on Imagenet database of natural images for the content loss) on microscopic images is not a good idea.

<p align="center"><img src="https://github.com/Saurabh23/mSRGAN-A-GAN-for-single-image-super-resolution-on-high-content-screening-microscopy-images./blob/master/thesis_scripts/images/dist.JPG" height="300" width="750" /></p>



# Solution:

I use the same architecture as SRGAN with the exception that 

**1]** Instead of using a pre-trained VGG19 model on Imagenet for the content loss, I train a miniVGG19 from scratch on microscopic images to classify protein sub cellular localizations across 13 classes.

**2]** Perceptual loss function used is the weighted sum of MSE, content loss and adversarial loss. Weight parameters (alpha, beta) to adjust the importance given to MSE and content loss respectively.

<p align="center"><img src="https://github.com/Saurabh23/mSRGAN-A-GAN-for-single-image-super-resolution-on-high-content-screening-microscopy-images./blob/master/thesis_scripts/images/loss.JPG" height="350" width="750" /></p>


  [SRGAN]: <https://arxiv.org/abs/1609.04802>
  [Bengio et al.]: <https://arxiv.org/abs/1411.1792>
  [Azizpour et al.]: <https://arxiv.org/abs/1406.5774>
