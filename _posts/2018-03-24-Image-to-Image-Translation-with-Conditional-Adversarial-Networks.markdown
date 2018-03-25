---
layout: post
comments: true
title:  "Image-to-Image Translation with Conditional Adversarial Networks"
date:   2018-03-24
author: stat17_hb
categories: DeepLearning
tags:	deeplearning
cover:  "/assets/header_image3.jpg"
---

[Image-to-Image Translation with Conditional Adversarial Networks][paper]

<a href="https://raw.githubusercontent.com/stat17-hb/stat17-hb.github.io/master/assets/pix2pix/figure1.PNG" data-lightbox="pix2pix" data-title="figure1">
  <img src="https://raw.githubusercontent.com/stat17-hb/stat17-hb.github.io/master/assets/pix2pix/figure1.PNG" title="figure1">
</a>

## Abstract

 이 논문에서는 image to image translation 문제에 대한 general purpose solution으로 conditional adversarial networks을 사용했다. 여기서 general purpose solution이라고 한 이유는 cGAN이 input image에서 output image로의 mapping을 학습할 뿐만 아니라 이 mapping을 훈련시키는데 필요한 loss function도 학습하기 때문이다. 
 
 기존의 모델에서는 문제의 기본적인 setting이 predict pixels from pixels로 모두 같음에도 불구하고, 각각의 문제마다 서로 다른 loss function이 필요했고, 이를 위한 노동(논문에서는 parameter tweaking, hand-engineering이라는 용어 사용)을 해야 했다(mapping fuction을 training 할 때도 노동이 필요했음). Figure1에서 볼 수 있는 것처럼 cGAN(conditional GAN)을 사용하여 이런 수고를 하지 않고도 다양한 문제를 같은 알고리즘으로 풀 수 있게 되었다.


## 1. Introduction

이 논문에서는 image to image translation 문제를 language translation 문제처럼 접근했다. 즉, 충분한 training data를 통해 하나의 image의 representation을 다른 representation으로 변환하는 방식을 배우도록 한 것이다.

기존에는 CNN을 사용해서 이러한 문제를 해결하려고 했는데, CNN은 learning process는 자동적이지만 효과적인 loss function을 설계하기 위한 수작업이 필요하다는 문제점이 있었다. 즉, loss function을 설계할 때 어떤 것을 minimize하고 싶은지 일일이 정해줘야 했다는 것이다. 나이브한 접근법으로 loss function을 predicted와 ground truth사이의 유클리드 거리로 설정하면 blurry한 image를 얻게 된다. 왜냐하면 유클리드 거리는 모든 plausible output들을 averaging하는 방식으로 minimize 되기 때문이다. loss function을 잘 정의해서 sharp하고 realistic한 이미지를 만들도록 하는 것이 여전히 문제로 남아있고, 이것은 expert knowledge가 필요한 부분이다.

'realistic image를 만들어라' 같은 high level에서의 목표만 정해주면 자동으로 loss function을 학습하는 방식으로 작동하는 모델이 있다면 좋을 것이다. 이런 생각이 최근의 GAN 모델들에서 구현되었다.

+ [GAN (오리지널)][23]
+ [Deep generative image models using a laplacian pyramid of adversarial networks][12]
+ [Unsupervised representation learning with deep convolutional generative adversarial networks (DCGAN)][43]
+ [Improved techniques for training GANs, T.Salimans et al., 2016][51]
+ [Energy-based generative adversarial network, J. Zhao et al., 2016][62]

GAN은 output image가 real인지 fake인지 판별(D)하기 위한 loss를 학습하면서, 동시에 이 loss를 minimize하기 위한 generative model(G)을 훈련시킨다. 여기서 blurry image는 분명히 fake로 판별될 것이기 때문에 GAN을 통해 학습을 하면 blurry image는 용인되지 않을 것이다. GAN은 데이터에 맞는 loss를 학습하기 때문에 기존에는 매우 다른 종류의 loss function이 필요했던 문제들에 공통적으로 적용할 수 있다.

이 논문에서는 conditional setting에서의 GAN모델을 탐색했다. cGAN은 input image에 condition을 주고 그에 상응하는 output image를 만들어내는 것이다.

이 논문의 contribution은 다음의 두 가지이다.

+ Primary : 다양한 문제에 대해 cGAN이 reasonable results를 만들어낸다는 것을 보이는 것

+ Secondary : 좋은 결과를 얻기에 충분한 단순한 프레임워크를 만들고, 몇 가지 중요한 구조적 선택들의 효과를 분석하는 것

저자가 공개한 코드는 [여기][code]에 있다.


## 2. Related work

**Structured losses for image modeling**

+ Image to Image translation 문제는 per-pixel classification/ regression으로 자주 공식화된다(fomulated).

+ 이런 fomulation은 output space를 input image가 주어졌을 때 각각의 output 픽셀이 다른 픽셀들과 조건부 독립으로 여겨진다는 점에서 "*unstructured*"로 취급한다.

+ cGAN은 이와달리 *structured loss*를 학습한다. structured loss는 output의 joint configuration을 penalize한다.








[paper]: https://phillipi.github.io/pix2pix/
[23]: https://arxiv.org/abs/1406.2661
[12]: https://arxiv.org/abs/1506.05751
[43]: https://arxiv.org/abs/1511.06434
[51]: https://arxiv.org/abs/1606.03498
[62]: https://arxiv.org/abs/1603.08511
[code]: https://github.com/phillipi/pix2pix
