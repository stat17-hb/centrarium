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

<a href="https://raw.githubusercontent.com/stat17-hb/stat17-hb.github.io/master/assets/pix2pix/figure1.PNG" data-lightbox="figure1" data-title="figure1">
  <img src="https://raw.githubusercontent.com/stat17-hb/stat17-hb.github.io/master/assets/pix2pix/figure1.PNG" title="figure1">
</a>

# Abstract

 이 논문에서는 image to image translation 문제에 대한 general purpose solution으로 conditional adversarial networks을 사용했다. 여기서 general purpose solution이라고 한 이유는 cGAN이 input image에서 output image로의 mapping을 학습할 뿐만 아니라 이 mapping을 훈련시키는데 필요한 loss function도 학습하기 때문이다. 
 
 기존의 모델에서는 문제의 기본적인 setting이 predict pixels from pixels로 모두 같음에도 불구하고, 각각의 문제마다 서로 다른 loss function이 필요했고, 이를 위한 노동(논문에서는 parameter tweaking, hand-engineering이라는 용어 사용)을 해야 했다(mapping fuction을 training 할 때도 노동이 필요했음). Figure1에서 볼 수 있는 것처럼 cGAN(conditional GAN)을 사용하여 이런 수고를 하지 않고도 다양한 문제를 같은 알고리즘으로 풀 수 있게 되었다.


# 1. Introduction

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


# 2. Related work

**Structured losses for image modeling**

+ Image to Image translation 문제는 per-pixel classification/ regression으로 자주 표현된다.

+ 이런 fomulation은 output space를 input image가 주어졌을 때 각각의 output 픽셀이 다른 픽셀들과 조건부 독립으로 여겨진다는 점에서 "*unstructured*"라고 할 수 있다.

+ cGAN은 이와달리 *structured loss*를 학습한다. structured loss는 output의 joint configuration을 penalize한다.

=> 이 부분 설명 필요
joint configuration이 무엇이고, 그것을 penalize한다는 것이 무엇인지

+ 상당수의 문헌에서는 structured loss를 다음과 같은 방법들로 다루고 있다.
	+  conditional random fields
	+  SSIM metric
	+  feature matching
	+  nonparametric losses
	+  convolutional pseudo-prior
	+  losses based on matching covariance statistics

+ cGAN은 output과 target 사이의 어떤 가능한 structure 차이도 다 penalize할 수 있다는 점에서 이와 다르다.

=> 여기서 target이 어떤 걸 말하는거지?

**Conditional GANs**

+ 이 논문이 GAN에 conditional setting을 처음 적용한 것은 아니다. 기존의 연구에서도 discrete label, text, image에 대한 conditioned GAN이 사용된 것들이 있다.

+ 다음과 같은 작업에서 conditional GAN을 image에 적용하였다.

	+ image prediction from a normal map
	+ future frame prediction
	+ product photo generation
	+ and image generation from sparse annotations

+ 다른 몇몇 논문들에서도 GAN을 image-to-image mapping에 사용하였지만, 단지 unconditional하게 GAN을 사용하였다. 즉, output이 input에 조건화 되게 하기 위해 L2 regression 같은 다른 term들에 의존했다.

+ 이 논문들은 다음과 같은 부분에서 인상적인 결과를 보였다.

	+ inpainting
	+ future state prediction
	+ image manipulation guided by user constraints
	+ style transfer
	+ superresolution

+ 위의 연구들에서 사용한 각각의 방법은 특정 용도에 맞게 만들어졌는데, 이 논문의 방법은 application-specific하지 않다는 점에서 위의 연구들과 다르다.

+ 또한, 이 논문에 적용된 방법은 generator와 discriminator에 대한 몇 가지 구조적 선택에서 기존의 연구와는 차이가 있다.

	+ generator에는 **U-Net** 기반 구조를 사용
		
	+ discriminator에는 convolutional "**PatchGAN**" classifier 사용 - image patch scale에서만 structure를 penalize

	+ U-Net과 PatchGAN이 무엇인지는 3.2. Network architectures에서 설명됨

# 3. Method

GAN은 random noise vector z로부터 output image로의 mapping을 학습하는 모델이다. 반면에 conditional GAN은 observed image x와 random noise vector z로부터 y로의 mapping을 학습한다. 

<a href="https://raw.githubusercontent.com/stat17-hb/stat17-hb.github.io/master/assets/pix2pix/figure2.PNG" data-lightbox="figure2" data-title="figure2">
  <img src="https://raw.githubusercontent.com/stat17-hb/stat17-hb.github.io/master/assets/pix2pix/figure2.PNG" title="figure2">
</a>

Figure2는 edges->photo로 mapping하는 cGAN의 training 과정을 보여준다. discriminator(D)는 fake(genertor(G)에 의해 만들어진 image)와 real {edge, photo} 튜플을 구분하는 학습을 한다. G는 D를 속이기 위한 학습을 한다. unconditional GAN과는 다르게 G와 D 모두 input edge map(목적함수에서 x로 표현됨)을 관측한다. unconditional GAN에서는 random noise z만 G에 input으로 들어갔다.

## 3.1. Objective

cGAN의 목적함수는 다음과 같이 표현된다:

$$L_{cGAN}(G, D) = E_{x,y}[logD(x,y)] \\ \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad +E_{x.z}[log(1-D(x,G(x,z)))] \quad \quad \quad \quad \quad \quad (1)$$

여기서 x는 edge map, z는 random noise, y는 real image이다. G는 이 목적함수를 최소화하려고 하고, 반대로 적대적인 D는 목적함수를 최대화하려고 한다.

즉, 
$$G^*=arg\;\underset{G}{min} \underset{D}{max}L_{cGAN}(G,D)$$ 
이다.

D를 conditioning하는 것의 중요성을 테스트하기 위해서, D가 x를 관측하지 않은 unconditional variant와 비교했다:

$$L_{cGAN}(G, D) = E_{x,y}[logD(x,y)] \\ \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad +E_{x.z}[log(1-D(G(x,z)))] \quad \quad \quad \quad \quad \quad (2)$$

선행연구([D. Pathak, P. Krahenbuhl, J. Donahue, T. Darrell, and A. A. Efros. Context encoders: Feature learning by inpainting.
CVPR, 2016.][42])에서는 GAN의 목적함수에 L2 distance같은 전통적인 loss를 섞는것이 더 좋다는 것을 발견했다. D가 해야하는 일은 기존과 같지만, G는 D를 속이려고 할 뿐만 아니라 L2의 관점에서 ground truth(real image를 말하는 듯 함)에 가까운 image를 출력해야 한다.

image blurring을 줄여주는 L1 distance도 사용하여 탐색했다:

$$L_{L1}(G)=E_{x,y,z}[||y-G(x,z)||] \quad \quad \quad \quad \quad \quad \quad \quad (3)$$

최종적인 목적함수는

$$G^*=arg\;\underset{G}{min} \underset{D}{max} L_{cGAN}(G,D)+ \lambda L_{L1}(G)$$

input에 x(e.g. edge map)와 random noise z를 같이 사용하는 이유는 z가 없어도 x에서 y로 가는 mapping을 학습하기는 하지만 stochastic하지 않고 deterministic한 결과만을 내놓기 때문이다. cGAN을 사용한 선행연구들에서도 이런점을 인지하고 x와 함께 Guassian random noise를 input으로 넣어왔다. 하지만 이 논문에서는 초기 실험에서 이러한 기법이 효과적이라는 증거를 찾기 못했고([다른 연구][39]에서도 이런 결과가 나온 경우가 있었다고 한다.), 최종 모델에서 noise를 dropout의 형태로만 제공했다. training과 test를 할 때 generator의 몇몇 layer에만 적용했다고 한다. 결과적으로 dropout noise를 추가한 것이 stochastic한 결과를 얻는게 크게 도움이 되지는 않았다. cGAN이 highly stochastic한 output을 만들어 내도록 설계하는 것이 앞으로의 과제로 남아있다.

### Dropout 이란?

<a href="https://raw.githubusercontent.com/stat17-hb/stat17-hb.github.io/master/assets/pix2pix/dropout1.png" data-lightbox="dropout1" data-title="dropout1">
  <img src="https://raw.githubusercontent.com/stat17-hb/stat17-hb.github.io/master/assets/pix2pix/dropout1.png" title="dropout1">
</a>

+ 신경망 전체를 다 학습시키지 않고 일부 노드만 무작위로 골라 학습시키는 기법
+ 전체 weight들을 모두 사용하지 않고 일부만 사용하는 것

<a href="https://raw.githubusercontent.com/stat17-hb/stat17-hb.github.io/master/assets/pix2pix/dropout2.png" data-lightbox="dropout2" data-title="dropout2">
  <img src="https://raw.githubusercontent.com/stat17-hb/stat17-hb.github.io/master/assets/pix2pix/dropout2.png" title="dropout2">
</a>

+ 학습하는 중간중간 일정 비율로 노드들을 무작위로 골라 출력을 0으로 만들어 신경망의 출력을 계산
+ 드롭아웃을 적용하면 학습되는 노드와 가중치들이 매번 달라져서 신경망이 과적합에 빠지는 것을 효과적으로 예방할 수 있다고 함

<a href="https://raw.githubusercontent.com/stat17-hb/stat17-hb.github.io/master/assets/pix2pix/dropout3.png" data-lightbox="dropout3" data-title="dropout3">
  <img src="https://raw.githubusercontent.com/stat17-hb/stat17-hb.github.io/master/assets/pix2pix/dropout3.png" title="dropout3">
</a>

+ 위의 슬라이드에서는 dropout이 어떻게 효과를 내는지 비유적으로 설명
+ 너무 많은 전문가가 있으면 오히려 판단에 장애요소로 작용할 수 있음
+ '사공이 많으면 배가 산으로 간다'

위의 슬라이드는 김성훈 교수님의 [모두를 위한 머신러닝/딥러닝 강의 깃허브][모두를 위한 딥러닝 깃허브]에서 가져옴

## 3.2. Network architectures

<a href="https://raw.githubusercontent.com/stat17-hb/stat17-hb.github.io/master/assets/pix2pix/figure3.PNG" data-lightbox="figure3" data-title="figure3">
  <img src="https://raw.githubusercontent.com/stat17-hb/stat17-hb.github.io/master/assets/pix2pix/figure3.PNG" title="figure3">
</a>

이 논문에서는 [Unsupervised representation learning with deep convolutional generative adversarial networks][43]에 있는 generator와 discriminator architectures를 사용했다.

generator와 discriminator 모두 [convolution-BatchNorm-ReLu][28] 형태의 모듈을 사용했다.

### 3.2.1 Generator with skips

image to image translation 문제에서 feature를 정의하는 것은 고해상도 input 그리드를 고해상도 output 그리드에 mapping하는 것이다. 우리가 고려하는 문제에 대해, 입력과 출력은 표면적인 형태는 다르지만 둘 다 동일한 기본 구조를 가진 랜더링이다. 따라서 입력에 있는 구조가 출력에 있는 구조와 대략적으로 정렬되어 있다. 이런한 것을 고려하여 generator architecture를 설계했다.

이전에 많은 연구들에서 이 문제를 풀기 위해 encoder-decoder network를 사용했다. 이런 네트워크 구조에서는 input이 레이어들을 거치면서 bottleneck layer에 도달할 때까지 downsample 되고, 그 이후에는 upsample 된다. image translation문제를 풀때는 많은 양의 low-level information이 input과 output에서 공유되는데, 이 information을 직접적으로 전달하는 것이 바람직하다. 즉, 모든 레이어를 거치지 않고 bottleneck 레이어를 기준으로 대응되는 레이어지 직접 전달하는 것이다. 이것을 skip connection 이라 하고, Figure 3의 U-Net이 이 구조를 나타낸 것이다.

### 3.2.2 Markovian discriminator (PatchGAN)


## 3.3. Optimization and inference



# 4. Experiments


## 4.1. Evaluation metrics


## 4.2. Analysis of the objective function


## 4.3. Analysis of the generator architecture


## 4.4. FromPixelGANs to PatchGANs to ImageGANs

## 4.5. Perceptual validation

## 4.6. Semantic segmentation

## 4.7. Communitydriven Research

# 5. Conclusion

[paper]: https://phillipi.github.io/pix2pix/
[23]: https://arxiv.org/abs/1406.2661
[12]: https://arxiv.org/abs/1506.05751
[43]: https://arxiv.org/abs/1511.06434
[51]: https://arxiv.org/abs/1606.03498
[62]: https://arxiv.org/abs/1603.08511
[code]: https://github.com/phillipi/pix2pix
[42]: https://arxiv.org/abs/1604.07379
[39]: https://arxiv.org/abs/1511.05440
[모두를 위한 딥러닝 깃허브]: https://hunkim.github.io/ml/
[28]: https://arxiv.org/abs/1502.03167