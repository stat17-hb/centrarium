---
layout: post
comments: true
title:  "Ridge Regression"
date:   2018-04-04
author: stat17_hb
categories: Statistics
tags: Ridge Regression
cover:  "/assets/header_image3.jpg"
---

# 1. Intro

원래 Ridge regression은 shrinkage method로써 만들어지지 않았다고 한다. 처음에는 linear regression에서 발생하는 multicollinearity(다중공선성: 독립변수들 간의 강한 상관관계) 문제를 해결하기 위한 방법으로써 고안되었다. 독립변수 x들의 correlation이 크면 $(X'X)^{-1}$가 불안정하게 된다. 여기서 불안정하다는 말의 의미는 X가 조금만 바뀌어도 $X'X$가 많이 바뀐다는 것이다. 결과적으로 이러한 계산의 불안정성 때문에 OLS(Ordinary Least Square) 추정값이 더 이상 BLUE(Best Linear Unbiased Estimator)가 되지 않는다. 만약 완전한 multicollinearity(독립변수들 간의 정확한 직선관계 = linearly dependent)가 있으면 $X'X$가 full rank가 아니게 되어 $(X'X)^{-1}$를 구할 수 없게 된다.

어쨌든, 사람들은 Ridge regression을 약간의 bias를 허용하는 대신 model의 variance를 크게 줄이는데 사용할 수 있다는 것을 알게 되었다. 독립변수들의 차원이 커지면 unbiased estimator가 biased estimator가 prediction accuracy(test MSE)관점에서는 항상 좋은 것은 아니며 bias가 너무 크지 않다면 오히려 biased estimator가 더 efficient한 경우가 생긴다.

[참고 - James-Stein Estimatior][js]

# 2. Linear regression

아마도 Linear regression은 MLE(Maximum Likelihood Estimator)에 기반한 가장 널리 사용되는 estimation 기법일 것이다.

linear model

$$y=X\beta+\epsilon \quad \quad \quad \quad \quad \quad \quad \quad \quad (1)$$

으로 부터 n차원의 벡터 $y=(y_1, y_2, \cdots, y_n)'$를 관측했다고 하자.

여기서 X는 $n \times p$ matrix이고, $\beta$는 p차원 벡터이다.

$\epsilon$(noise vector)은 n차원 벡터인데, 각 요소들은 uncorrelated(=independent)이고 $\sigma^2$의 분산을 갖는다.

$$\epsilon \sim (0, \sigma^2I) \quad \quad \quad \quad \quad \quad \quad \quad \quad (2)$$

여기서 $I$는 $n \times n$ identity matrix이다.

종종 $\epsilon$이 multivariate normal을 따른다고 가정하는데

$$\epsilon \sim N_n(0, \sigma^2I) \quad \quad \quad \quad \quad \quad \quad \quad \quad (3)$$

이어질 내용의 대부분에서는 이 가정이 필요하지 않다.

1800년대 초의 가우스(Gauss)와 르장드르(Legendre) 까지 거슬러 올라가는 최소 제곱 추정량(least squares estimate) $\hat{\beta}$은 total sum of squared errors를 minimize함으로써 구해진다.

$$\hat{\beta}=arg \; \underset{\beta} min \{||y-X\beta||^2\} \quad \quad \quad \quad \quad \quad \quad \quad \quad (4)$$

결과적으로 $\hat{\beta}$는 다음 식을 이용하여 구할 수 있다.

$$\hat{\beta}=(X'X)^{-1}X'y \quad \quad \quad \quad \quad \quad \quad \quad \quad (5)$$

$\hat{\beta}$는 $\beta$에 대한 unbiased estimator이고 $\sigma^2(X'X)^{-1}$를 covariace matrix로 갖는다.

$$\hat{\beta} \sim (\beta, \; \sigma^2(X'X)^{-1}) \quad \quad \quad \quad \quad \quad \quad \quad \quad (6)$$

(2)에서 처럼 Normal을 따른다고 가정하면 $\hat{\beta}$는 $\beta$에 대한 MLE이다.

linear model의 가장 큰 장점은 추정해야 할 unknown parameter를 독립변수의 개수인 p(혹은 \sigma^2까지 더하서 p+1)개 까지 줄여준다는 것이다. p가 작은 경우에는 이 방법이 효과적이다. 하지만 p가 큰 경우에는 unbiased estimation에 한계가 있다. 

# 3. Ridge regression

Ridge regression 모델을 fitting하기 전에 먼저 해야할 일이 있다. $(1)$에서 $X$의 각 column들을 평균 0, 제곱합 1로 변환하는 것이다. 이렇게 하면 

$$(X'X)_{ii}=1, \quad for \;i=1,2,\cdots,p$$

가 된다. 이렇게 하는 이유는 회귀계수 $\beta_1, \beta_2, ..., \beta_p$를 비교할 수 있는 스케일로 맞춰주기 위함이다.

$X$의 각 column들을 평균 0, 제곱합 1로 변환하는 방법은 다음과 같다.

$C_i$를 $X$의 i번째 column이라고 하면,

$$C_{i}' = {{C_i - mean(C_i)} \over {\sqrt{sum(C_i - mean(C_i))^2}}}$$



# 4. Reference

Bradley Efron, Tevor Hastie(2017), Computor Age Statistical Inference, Cambridge University Press.
나종화(2017). R 응용 회귀분석, 자유아카데미.


[js]: https://en.wikipedia.org/wiki/James%E2%80%93Stein_estimator