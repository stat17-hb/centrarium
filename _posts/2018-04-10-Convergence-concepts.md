---
layout: post
comments: true
title:  "Convergence concept"
date:   2018-04-10
author: stat17_hb
categories: Statistics
tags: Convergence
cover:  "/assets/header_image3.jpg"
---

# Convergence Concepts

infinite sample size라는 개념이 이론적으로만 존재하는 것이기는 하지만, finite sample case에 대한 유용한 approximation을 제공해준다. 이는 limit 개념을 활용할 때 expression이 단순해지는 경우가 많기 때문이다.

_ _ _

## Convergence in Probability

+ Weaker type 중 하나이기 때문에 확인하기 쉽다.

### Definition

모든 $$\epsilon > 0$$에 대하여 다음이 성립하면 $$X_1, X_2, \cdots,$$의 random variable들의 sequence가 random variable $$X$$로 **converges in probability**한다고 한다.

$$\underset{n \rightarrow \infty} {lim} P(|X_n-X|\ge\epsilon)=0 \;\; or, equivalently, \quad \underset{n \rightarrow \infty} {lim}P(|X_n-X|< \epsilon)=1$$

+ 여기서 고려하는 random variable들은 iid 조건이 없다.

+ n이 커질때 $$X_n$$의 분포가 어떤 limiting distribution으로 converge하는 몇 가지 방법이 있다.

+ 보통 limiting random variable이 상수이고, sequence에서 random variable이 sample mean인 상황을 고려한다. Convergence in Probability에서 가장 유명한 결과는 **Weak Law of Large Numbers(WLLN)**이다.

_ _ _

### Weak Law of Large Numbers

$$X_1, X_2, ...$$를 $$E(X_i)=\mu$$이고 $$Var(X_i)=\sigma^2<\infty$$인 **iid** random variables라 하자. $$\bar{X}=(1/n)\Sigma_{i=1}^n X_i$$로 정의하면 모든 $$\epsilon >0$$에 대하여

$$\underset{n \rightarrow \infty} {lim} P(|X_n-\mu|<\epsilon)=1$$

이다. 즉, $$\bar{X}_n$$ converges in probability to $$\mu$$

---

$$[Proof]$$ Chebychev's inequality이용

for every $$\epsilon > 0$$,

$$P(|X_n-\mu|\ge\epsilon)$$

