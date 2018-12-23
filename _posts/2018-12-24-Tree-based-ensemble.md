---
layout: post
comments: true
title:  "Tree-based enesemble"
date:   2018-12-24
author: stat17_hb
categories: ML
tags:	ML
cover:  "/assets/header_image3.jpg"
---

이 글은 고려대학교 강필성 교수님의 2018년 2학기 비즈니스 어낼리틱스 강의 내용을 바탕으로 작성되었습니다.

# Random Forests

## Intro

랜덤 포레스트는 트리들의 상관성을 제거하는(decorrlate) 단순한 방법으로 배깅 트리보다 더 좋은 성능을 보여준다. 트리들의 상관성을 제거하는 이유는 모델의 다양성을 증가시키기 위함이다. 랜덤 포레스트에서는 모델의 다양성(diversity)을 증가시키기 위해 두 가지 방법을 사용한다. 하나는 배깅이고, 다른 하나는 각 분할(split)에서 사용할 예측변수들을 랜덤하게 선택하는 것이다. 배깅은 부트스트랩(bootstrap)을 통해 각각의 트리를 만들때 사용되는 인스턴스의 다양성을 증가시킨다. 각 분할에서 사용할 예측변수들을 랜덤하게 선택하는 것은 모델에 사용되는 변수의 다양성을 증가시키는 효과를 준다. 이 글에서는 모델의 다양성을 증가시키기 위해 랜덤포레스트가 사용하는 방법들을 살펴보고, Out of Bag, Variable importance와 관련된 내용들을 정리해보고자 한다.

---

## Algorithm

랜덤 포레스트 알고리즘([\[1\]][1] p.588)은 다음과 같다:

<a href="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/rfalgo.PNG?raw=true" data-lightbox="rfalgo" data-title="rfalgo">
  <img src="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/rfalgo.PNG?raw=true" title="rfalgo" width="300">
</a>

## Diversity in Ensemble Model

모델의 다양성을 증가시키기 위한 두 가지 방법에 대해 더 자세하게 살펴보자. 배깅의 핵심 아이디어는 모델의 분산을 줄이기 위해 분산이 크지만 편향(bias)이 작은 모델들을 많이 만들어서 평균을 내는 것이다. 트리 알고리즘은 모델의 분산이 큰 알고리즘 중 하나이기 때문에 평균을 내는 것이 매우 효과적이다. 게다가, 배깅을 통해 생성된 트리 각각은 동일한 분포를 따르기 때문에(identically distributed) B개 트리들의 평균(sample average)의 기댓값(expectation)은 각각의 트리의 기댓값과 같다. 이 말은 배깅된 트리들의 편향은 개별 트리의 편향과 같고, 따라서 모델을 개선시킬 수 있는 유일한 방법은 분산을 줄이는 것이라는 의미이다. 반면, 부스팅에서는 트리들이 적응적인(adaptive) 방식으로 편향을 제거하면서 커지기 때문에 동일한 분포를 따르지 않는다.

만약 B개의 트리들이 독립적이고 동일한 분포를 따르고(i.i.d.; independently and identically distributed), 각 트리들의 분산이 $$\sigma^2$$이면 B개 트리들의 평균의 분산은 $$\frac{1}{B} \sigma^2$$이 될 것이다. 하지만 B개의 트리들이 독립적이지 않다고 하면, B개 트리들의 평균의 분산은 $$\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$$이 된다. 여기서 $$\rho$$는 트리 쌍들의 상관계수이다. B가 커지면 두 번째 항이 매우 작아지게 되겠지만 여전히 첫 번째 항이 남아있게 된다. 즉, 평균을 내는 것을 통해 분산이 얼마나 줄어들지는 트리 쌍들의 상관계수에 따라 달라진다. 따라서 트리 쌍들의 상관계수를 줄일 수 있다면 앙상블 모델 전체의 분산도 줄일 수 있을 것이다. 

배깅에서는 각 분할에 p개의 예측변수를 모두 사용한다. 만약 데이터에 매우 강한 설명변수가 하나 있고, 다수의 적당히 강한 설명변수가 있다고 해보자. 그러면 모든 부트스트랩 데이터세트의 맨 위 분할(top split)에 가장 강한 설명 변수가 사용될 것이다. 따라서 배깅된 트리들은 모두 서로 상당히 유사할 것이고, 결과적으로 단일 트리에 비해 모델의 분산을 크게 줄이지 못할 것이다. 랜덤포레스트는 각 분할에서 랜덤하게 선택된 p보다 작은 m개의 예측변수를 사용하는 것을 통해 트리들의 상관성을 줄인다. 이를 통해 앙상블 모델의 분산을 배깅에서보다 크게 줄 일 수 있게 된다.

일반적으로 m의 기본 값은 분류문제에서는 $$\sqrt{p}$$, 회귀문제에서는 $$p/3$$이다(랜덤 포레스트를 만든 Leo Breiman과 Adele Cutler가 추천하는 값이다). 랜덤 포레스트가 m에 민감하지 않기 때문에 굳이 m에 대한 미세조정(fine tuning)을 할 필요는 없다고 한다 [[2]](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-3).

파이썬을 이용한 예제를 통해 이에 대해 살펴보자.

{% highlight python %}
# Load libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor

# Load data
from sklearn.datasets import load_boston
boston = load_boston()
boston.DESCR
X_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
y_boston = pd.DataFrame(boston.target, columns=["MEDV"])
all_boston = pd.concat([X_boston, y_boston], axis=1)
all_boston.head()
{% endhighlight %}

<a href="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/boston.PNG?raw=true" data-lightbox="boston" data-title="boston">
  <img src="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/boston.PNG?raw=true" title="boston" width="400">
</a>

{% highlight python %}
# Split training and test set
N = len(all_boston)
ratio = 0.75
random.seed(0)
idx_train = np.random.choice(np.arange(N), np.int(ratio * N), replace=False)
idx_test = list(set(np.arange(N)).difference(idx_train))

X_train = all_boston.iloc[idx_train, 0:13]
y_train = all_boston.iloc[idx_train, 13]
X_test = all_boston.iloc[idx_test, 0:13]
y_test = all_boston.iloc[idx_test, 13]
{% endhighlight %}

{% highlight python %}
# Tuning number of features at each split
cv_error=[]
cv_mse=[]
N = len(all_boston)
random.seed(0)
idx_fold = np.random.choice(np.arange(5), N, replace=True)
for n_features in np.array(range(13))+1:
    temp_mse = []
    temp_error = []
    for fold in range(5):
        print(fold+1, "th fold ", "n_features=",n_features)
        idx_test = idx_fold==fold
        idx_train = idx_fold!=fold
        X_train = all_boston.iloc[idx_train, 0:13]
        y_train = all_boston.iloc[idx_train, 13]
        X_test = all_boston.iloc[idx_test, 0:13]
        y_test = all_boston.iloc[idx_test, 13]
    
        rf = RandomForestRegressor(n_estimators=500, max_features=n_features, 
                                   random_state=0, n_jobs=-1, oob_score=True)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        temp_mse.append(np.mean((rf_pred-np.array(y_test).flatten())**2))
        temp_error.append(1-rf.score(X_test,y_test))
    cv_mse.append(np.mean(temp_mse))
    cv_error.append(np.mean(temp_error))
{% endhighlight %}

{% highlight python %}
# Draw plot
n_features = range(1,14)
plt.style.use('ggplot')
plt.plot(n_features, cv_mse, label="CV Error")
plt.xlabel('Number of features at each split')
plt.ylabel('Error')
plt.legend()
plt.show()
{% endhighlight %}

<a href="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/nfeatures.png?raw=true" data-lightbox="nfeatures" data-title="nfeatures">
  <img src="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/nfeatures.png?raw=true" title="nfeatures" width="300">
</a>


{% highlight python %}

{% endhighlight %}

[1]: https://web.stanford.edu/~hastie/ElemStatLearn/
