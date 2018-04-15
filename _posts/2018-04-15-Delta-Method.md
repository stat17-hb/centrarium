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

Delta Method는 중심극한정리(Central Limit Theorem)의 일반화이다. 중심극한정리에서는 limit normal distribution을 갖는 표준화된 확률변수를 다루었다. 하지만 때로는 확률변수 자체의 분포보다는 확률변수의 함수의 분포에 관심이 있을 수 있다. 이때 유용하게 사용될 수 있는 것이 Delta Method이다.

_ _ _

## Theorem - Delta Method

$$X_n$$을 $$sqrt{n}(X_n-\theta) \overset{d}{\to} N(0, \sigma^2)$$인 확률변수들의 sequence라고 하자. 

함수 $$g(x)$$가 $$\theta$$에서 미분가능하고 $$g'(\theta)\ne$$0이라고 가정하면

$$\sqrt{n}[g(X_n)-\theta] \overset{d}{\to}N(0, \sigma^2[g'(\theta)]^2)$$

_ _ _

## Proof - Delta Method

$$g(X_n)$$의 $$X_n=\theta$$ 주변에서의 Talyor expansion은

$$g(X_n) \approx g(\theta)+ (X_n-\theta)g'(\theta) + Remainder$$

이다.

여기서 Remainder는 $$X_n \to \theta$$일때 0으로 가는데, $$X_n \overset{p}\to \theta$$이기 때문에 $$Remainder \overset{p}\to 0$$이 된다.

$$X_n \overset{p}\to \theta$$에 대한 증명은 다음과 같다.

$$P(|X_n-\theta|<\epsilon)=P(|\sqrt{n}(Y_n)-\theta|<\sqrt{n}\epsilon)$$

$$\underset{n \to \infty}{lim} P(|X_n-\theta|<\epsilon)= \underset{n \to \infty}{lim} P(|\sqrt{n}(Y_n)-\theta|<\sqrt{n}\epsilon) = P(|Z|<\infty)=1$$

$$where \quad Z \sim N(0, \sigma^2)$$

$$\therefore \quad X_n \overset{p} \to \theta$$






# Reference

George Casella, Roger L.Berger(2001), Statistical Inference 2nd E, Duxbury Press