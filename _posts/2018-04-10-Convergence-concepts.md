---
layout: post
comments: true
title:  "Convergence concepts"
date:   2018-04-10
author: stat17_hb
categories: Statistics
tags: Convergence
cover:  "/assets/header_image3.jpg"
---

infinite sample size라는 개념이 이론적으로만 존재하는 것이기는 하지만, finite sample case에 대한 유용한 approximation을 제공해준다. 이는 limit 개념을 활용할 때 expression이 단순해지는 경우가 많기 때문이다.

_ _ _

# Convergence in Probability

+ Weaker type 중 하나이기 때문에 확인하기 쉽다.

## Definition

모든 $$\epsilon > 0$$에 대하여 다음이 성립하면 $$X_1, X_2, \cdots,$$의 random variable들의 sequence가 random variable $$X$$로 **converges in probability**한다고 한다.

$$\underset{n \rightarrow \infty} {lim} P(|X_n-X|\ge\epsilon)=0 \;\; or, equivalently, \quad \underset{n \rightarrow \infty} {lim}P(|X_n-X|< \epsilon)=1$$

+ 여기서 고려하는 random variable들은 iid 조건이 없다.

+ n이 커질때 $$X_n$$의 분포가 어떤 limiting distribution으로 converge하는 몇 가지 방법이 있다.

+ 보통 limiting random variable이 상수이고, sequence에서 random variable이 sample mean인 상황을 고려한다. Convergence in Probability에서 가장 유명한 결과는 **Weak Law of Large Numbers(WLLN)**이다.

_ _ _

# Weak Law of Large Numbers

$$X_1, X_2, ...$$를 $$E(X_i)=\mu$$이고 $$Var(X_i)=\sigma^2<\infty$$인 **iid** random variables라 하자. $$\bar{X}=(1/n)\Sigma_{i=1}^n X_i$$로 정의하면 모든 $$\epsilon >0$$에 대하여

$$\underset{n \rightarrow \infty} {lim} P(|X_n-\mu|<\epsilon)=1$$

이다. 즉, $$\bar{X}_n$$ converges in probability to $$\mu$$

---

$$[Proof]$$ **Chebychev's inequality**이용   

for every $$\epsilon > 0$$,

$$P(|X_n-\mu|\ge\epsilon)=P((\bar{X}_n-\mu)^2\ge \epsilon^2) \le \frac {E(\bar{X}_n-\mu)^2}{\epsilon^2}=\frac{Var(\bar{X}_n)}{\epsilon^2}=\frac{\sigma^2}{n\epsilon^2}$$

따라서, 

$$P(|\bar{X}_n-\mu|<\epsilon)=1-P(|X_n-\mu|\ge\epsilon)\ge 1-\frac{\sigma^2}{(n\epsilon^2)}\rightarrow 1,\;\; as\;\; n\rightarrow \infty$$

---

**Chebychev's inequality**

$$\mu=E(X)$$, $$\sigma^2=Var(X)$$이라 하자. 그러면,

$$P(|X-\mu|\ge t) \le \frac{\sigma^2}{t^2}$$

_ _ _

# Consistency

WLLN으로 설명할 수 있는 개념 중에 **consistency**가 있다. consitency는 동일한 sample quantity의 sequence가 n이 $$\infty$$로 갈 때 어떤 상수로 다가간다는 것이다.

$$\hat{\theta}_n \overset{p}{\to} \theta, \quad as \;\; n \to \infty$$

이면 $$\hat{\theta}_n$$을 $$\theta$$에 대한 consistent estimator라고 한다.

## Example (Consistency of $$S^2$$)

$$E(X_i)=\mu$$이고 $$Var(X_i)=\sigma^2<\infty$$인 iid random variable $$X_1, \cdots ,X_n$$의 sequence가 있다고 할 때,

$$S^2_n=\frac{1}{n-1}\Sigma_{i=1}^n (X_i - \bar{X}_n)^2=\frac{1}{n-1}\{\Sigma_{i=1}^n X_i^2 - n\bar{X}_n^2\}=\frac{n}{n-1}\{\frac{1}{n}\Sigma_{i=1}^n X_i^2 - \bar{X}_n^2\}$$

$$n \to \infty$$ 일 때, WLLN을 적용하여 $$ \frac{1}{n}\Sigma_{i=1}^n X_i^2 \overset{p}{\to} \sigma^2+\mu^2$$이고, $$ \bar{X}_n^2 \overset{p}{\to} \mu^2$$이므로

$$S^2_n\overset{p}{\to} \sigma^2$$

---

만약 $$X \sim N(\mu, \sigma^2)$$라 하면,

$$\frac{(n-1)S^2_n}{\sigma^2} \sim \chi^2(n-1)$$이므로 $$Var(S^2_n)=\frac{2\sigma^4}{n-1}$$

$$P(|S^2_n-\sigma^2|\ge \epsilon) \le \frac{E(S^2_n - \sigma^2)^2}{\epsilon^2}=\frac{Var(S^2_n)}{\epsilon^2}$$

이므로, $$S^2_n$$이 $$\sigma^2$$로 converge in probability하는 충분조건은 $$n \rightarrow  \infty$$일 때, $$Var(S^2_n) \rightarrow 0$$ 인 것이다.

$$n \to \infty$$일 때, $$\frac{Var(S^2_n)}{\epsilon^2} \to 0$$이므로

$$S^2_n$$ converge in probability to $$\sigma^2$$이다.



_ _ _

# Reference

George Casella, Roger L.Berger(2001), Statistical Inference 2nd E, Duxbury Press