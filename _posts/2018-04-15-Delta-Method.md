---
layout: post
comments: true
title:  "Delta Method"
date:   2018-04-15
author: stat17_hb
categories: Statistics
tags: Delta Method
cover:  "/assets/header_image3.jpg"
---

Delta Method는 중심극한정리(Central Limit Theorem)를 일반화한 개념이라고 할 수 있다. 중심극한정리에서는 normal distribution을 극한분포로 갖는 표준화된 확률변수를 다룬다. 하지만 때로는 확률변수 자체의 분포보다는 확률변수의 함수의 분포에 관심이 있을 수 있다. 이때 유용하게 사용될 수 있는 것이 Delta Method이다.

_ _ _

## Talyor Seires Expansion

Delta Method는 Talyor Seires Expansion을 통해 유도 된다.

$$g(X) \approx g(\mu) + (X-\mu)g'(\mu) + \frac{(X-\mu)^2g''(\mu)}{2}+ \cdots$$

$$X$$가 확률변수이고, $$E(X)=\mu\ne0$$라고 하자. $$g(x)$$가 r차 미분값을 가지고 있다고 할 때, 함수 $$g(\mu)$$를 추정하고 싶다면, 

1st order approximation을 통해 $$E[g(X)] \approx g(\mu)$$를 얻을 수 있고, 

2nd order approximation을 통해 $$Var[g(X)] \approx [g'(\mu)]^2Var(X)$$를 얻을 수 있다.

_ _ _

## Example

$$X_1, ..., X_n \overset{iid} {\sim} Bernoulli(p)$$라고 할 때, odds인 $$\frac{p}{1-p}$$를 추정하고 싶다고 하자. 보통 success probability $$p$$를 $$\hat{p}$$로 추정하고, odds에 대한 추정치를 $$\frac{\hat{p}}{1-\hat{p}}$$로 사용한다. 여기서 $$\frac{\hat{p}}{1-\hat{p}}$$의 variance 근사치를 구하고 싶을 때 delta method를 사용할 수 있다.

$$g(p)=\frac{p}{1-p}$$라 하면

$$g'(p)=\frac{1*(1-p)-p*(-1)}{(1-p)^2}=\frac{1}{(1-p)^2}$$

$$Var[g(X)] \approx [g'(\mu)]^2Var(X)$$이므로

$$Var(\frac{\hat{p}}{1-\hat{p}})=[\frac{1}{(1-p)^2}]^2 \frac{p(1-p)}{n} = \frac{p}{n(1-p)^3}$$

이다. 

_ _ _

이제 중심극한정리의 일반화된 형태로서의 Delta Method를 살펴보자.

## Theorem - Delta Method - Univariate Case

$$X_n$$을 $$\sqrt{n}(X_n-\theta) \overset{d}{\to} N(0, \sigma^2)$$인 확률변수들의 sequence라고 하자. 

함수 $$g(x)$$가 $$\theta$$에서 미분가능하고 $$g'(\theta)\ne$$0이라고 가정하면

$$\sqrt{n}[g(X_n-\theta)] \overset{d}{\to}N(0, \sigma^2[g'(\theta)]^2)$$

_ _ _

## Proof - Delta Method

$$g(X_n)$$의 $$X_n=\theta$$ 주변에서의 Talyor expansion은

$$g(X_n) \approx g(\theta)+ (X_n-\theta)g'(\theta) + Remainder$$

이다.

여기서 Remainder는 $$X_n \to \theta$$일때 0으로 가는데, $$X_n \overset{p}\to \theta$$이기 때문에 $$Remainder \overset{p}\to 0$$이 된다.

$$X_n \overset{p}\to \theta$$에 대한 증명은 다음과 같다.

---

$$P(|X_n-\theta|<\epsilon)=P(|\sqrt{n}(X_n-\theta)|<\sqrt{n}\epsilon)$$

$$\underset{n \to \infty}{lim} P(|X_n-\theta|<\epsilon)= \underset{n \to \infty}{lim} P(|\sqrt{n}(X_n-\theta)|<\sqrt{n}\epsilon) = P(|Z|<\infty)=1$$

$$where \quad Z \sim N(0, \sigma^2)$$

$$\therefore \quad X_n \overset{p} \to \theta$$

---

$$g(\theta)$$를 우변으로 넘기면

$$g(X_n) - g(\theta) \approx (X_n-\theta)g'(\theta) + Remainder$$

이고,

$$\sqrt{n}[g(X_n)-g(\theta)] \approx g'(\theta)\sqrt{n}(X_n-\theta)$$

으로 표현할 수 있다.

이제 

$$\sqrt{n}[g(X_n)-g(\theta)] \approx g'(\theta)\sqrt{n}(X_n-\theta)$$

에 **Slutsky's Theorem**을 적용하면

$$g'(\theta) \sqrt{n}(X_n-\theta) \overset{d}\to g'(\theta)Z, \quad \quad where \quad Z \sim N(0, \sigma^2)$$

이 되고, 따라서

$$\sqrt{n}[g(X_n)-g(\theta)]=g'(\theta)\sqrt{n}(X_n-\theta) \overset{d}\to N(0, \sigma^2[g'(\theta)]^2)$$

이 된다.

_ _ _

### 참고 - Slutsky's Theorem

만약 $$X_n \overset{d}{\to} X$$이고, 상수 a에 대하여 $$Y_n \overset{p}{\to} a$$이면

+ $$Y_nX_n \overset{d}{\to} aX$$

+ $$X_n \pm Y_n \overset{d}{\to} X+a$$

+ $$\frac{X_n}{Y_n} \overset{d}{\to} \frac{X}{a}$$

가 성립한다.

일반적으로는 위의 명제들이 성립하지 않지만 $$X_n$$와 $$Y_n$$ 중 하나가 상수로 converge in probability이면 성립하게 되는 것으로 이해할 수 있다.

---

#### Example - Normal approximation with estimated variance

$$X_1, \cdots, X_n \sim N(\mu, \sigma^2)$$이고, $$S^2=\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\bar{X_n})^2$$일 때,

$$\frac{\sqrt{n}(\bar{X_n}-\mu)}{S} \sim t_{(n-1)}$$이다.

이 때, $$n$$이 커지면 중심극한정리에 의해 $$N(0,1)$$로 분포수렴(converge in distribution)한다.

이를 Slutsky's Theorem을 통해 설명할 수 있는데,

$$\frac{\sqrt{n}(\bar{X_n}-\mu)}{S}=\frac{\frac{\sqrt{n}(\bar{X_n}-\mu)}{\sigma}}{\frac{S}{\sigma}}$$

에서 $$\frac{\sqrt{n}(\bar{X_n}-\mu)}{\sigma} \overset{d}{\to} N(0,1)$$이고, $$S \overset{p}{\to} \sigma$$이므로 $$\frac{S}{\sigma} \overset{p}{\to}1$$이 되어 Slutsky's Theorem에 의해 n이 커질 때 t분포가 표준정규분포로 수렴함을 보일 수 있다.
_ _ _

## Variance Stabilizing Transformation(분산 안정화 변환)

분산 안정화 변환은 Delta Method의 대표적인 활용 사례이다. Asymtotic variance가 모수에 의존하는 경우에 이를 상수로 바꿔주는 과정이라고 할 수 있다.

몇 가지 예를 들어 보면,

### (1) Poisson

$$X_1, \cdots, X_n \sim Poisson(\lambda)$$라고 할 때,

중심극한정리에 의해 $$\sqrt{(\bar{X}_n-\lambda)} \overset{d}\to N(0, \lambda)$$라고 할 수 있고,

함수 $$h(X)$$가 있을 때, Delta Method에 의해

$$\sqrt{(h(\bar{X}_n)-h(\lambda))} \overset{d}\to N(0, [h'(\lambda)]^2\lambda)$$가 성립한다.

그런데 여기서 분산이 모수인 $$\lambda$$에 의존하는 것을 볼 수 있다. 이를 상수로 만들어주고 싶은데, 그렇게 하기 위해서는 $$[h'(\lambda)]^2 \propto \frac{1}{\lambda}$$ 형태가 되도록하는 $$h(\lambda)$$를 찾아주면 된다.

$$[h'(\lambda)] \propto \frac{1}{\sqrt{\lambda}}$$가 되도록 하는 $$h(\lambda)$$의 형태 중 하나는 $$h(\lambda)=2\sqrt{\lambda}$$가 있다.

### (2) Exponential

$$X_1, \cdots, X_n \sim Exp(\lambda)$$($$\lambda$$는 scale parameter)라고 할 때,

중심극한정리에 의해 $$\sqrt{(\bar{X}_n-\lambda)} \overset{d}\to N(0, \lambda^2)$$라고 할 수 있고,

함수 $$h(X)$$가 있을 때, Delta Method에 의해

$$\sqrt{(h(\bar{X}_n)-h(\lambda))} \overset{d}\to N(0, [h'(\lambda)]^2\lambda^2)$$가 성립한다.

위에서와 같은 방식으로 $$[h'(\lambda)]^2 \propto \frac{1}{\lambda^2}$$ 형태가 되도록하는 $$h(\lambda)$$를 찾아주면 된다.

$$[h'(\lambda)] \propto \frac{1}{\lambda}$$가 되도록 하는 $$h(\lambda)$$의 형태 중 하나는 $$h(\lambda)= log(\lambda)$$가 있다.

### (3) Bernoulli

$$X_1, \cdots, X_n \sim Bernoulli(p)$$, $$\bar{X}_n= \frac{1}{n}\Sigma X_i = \hat{p}_n$$라고 할 때,

중심극한정리에 의해 $$\sqrt{n}(\hat{p}_n-p) \overset{d}\to N(0, p(1-p))$$이고, (여기서 사용된 중심극한정리의 형태를 DeMoivre-Laplace Theorem이라고도 한다.)

함수 $$h(X)$$가 있을 때, Delta Method에 의해

$$\sqrt{(h(\hat{p}_n)-h(p)} \overset{d}\to N(0, [h'(p)]^2p(1-p))$$가 성립한다.

$$h'(p) \propto \frac{1}{\sqrt{p(1-p)}}$$가 되도록하는 $$h(p)$$를 찾으려면 적분을 해야한다.

$$h(p)= \int^p_a \frac{1}{\sqrt{x(1-x)}}dx$$를 풀면 결과적으로

$$2sin^{-1}\sqrt{p}$$가 나온다.

_ _ _

## Second order Delta Method

Delta Method를 사용할 때, $$g'(\theta)=0$$인 경우를 생각해볼 수 있다.

이 경우에는 Taylor Series Expansion에서 1st order term이 사라지기 때문에 2nd order term으로 Delta Method를 전개하게 되고, 수렴하는 분포도 정규분포에서 카이제곱분포로 바뀌게 된다.

$$n[g(X_n)-g(\theta)] \overset{d}\to \sigma^2\frac{g''(\theta)}{2}\chi^2$$

_ _ _

# Reference

George Casella, Roger L.Berger(2001), Statistical Inference 2nd E, Duxbury Press