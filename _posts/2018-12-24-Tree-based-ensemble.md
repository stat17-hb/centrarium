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
  <img src="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/rfalgo.PNG?raw=true" title="rfalgo" width="400">
</a>

## Diversity in Ensemble Model

모델의 다양성을 증가시키기 위한 두 가지 방법에 대해 더 자세하게 살펴보자. 배깅의 핵심 아이디어는 모델의 분산을 줄이기 위해 분산이 크지만 편향(bias)이 작은 모델들을 많이 만들어서 평균을 내는 것이다. 트리 알고리즘은 모델의 분산이 큰 알고리즘 중 하나이기 때문에 평균을 내는 것이 매우 효과적이다. 게다가, 배깅을 통해 생성된 트리 각각은 동일한 분포를 따르기 때문에(identically distributed) B개 트리들의 평균(sample average)의 기댓값(expectation)은 각각의 트리의 기댓값과 같다. 이 말은 배깅된 트리들의 편향은 개별 트리의 편향과 같고, 따라서 모델을 개선시킬 수 있는 유일한 방법은 분산을 줄이는 것이라는 의미이다. 반면, 부스팅에서는 트리들이 적응적인(adaptive) 방식으로 편향을 제거하면서 커지기 때문에 동일한 분포를 따르지 않는다.

만약 B개의 트리들이 독립적이고 동일한 분포를 따르고(i.i.d.; independently and identically distributed), 각 트리들의 분산이 $$\sigma^2$$이면 B개 트리들의 평균의 분산은 $$\frac{1}{B} \sigma^2$$이 될 것이다. 하지만 B개의 트리들이 독립적이지 않다고 하면, B개 트리들의 평균의 분산은 $$\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$$이 된다. 여기서 $$\rho$$는 트리 쌍들의 상관계수이다. B가 커지면 두 번째 항이 매우 작아지게 되겠지만 여전히 첫 번째 항이 남아있게 된다. 즉, 평균을 내는 것을 통해 분산이 얼마나 줄어들지는 트리 쌍들의 상관계수에 따라 달라진다. 따라서 트리 쌍들의 상관계수를 줄일 수 있다면 앙상블 모델 전체의 분산도 줄일 수 있을 것이다. 

배깅에서는 각 분할에 p개의 예측변수를 모두 사용한다. 만약 데이터에 매우 강한 설명변수가 하나 있고, 다수의 적당히 강한 설명변수가 있다고 해보자. 그러면 모든 부트스트랩 데이터세트의 맨 위 분할(top split)에 가장 강한 설명 변수가 사용될 것이다. 따라서 배깅된 트리들은 모두 서로 상당히 유사할 것이고, 결과적으로 단일 트리에 비해 모델의 분산을 크게 줄이지 못할 것이다. 랜덤포레스트는 각 분할에서 랜덤하게 선택된 p보다 작은 m개의 예측변수를 사용하는 것을 통해 트리들의 상관성을 줄인다. 이를 통해 앙상블 모델의 분산을 배깅에서보다 크게 줄 일 수 있게 된다.

일반적으로 m의 기본 값은 분류문제에서는 $$\sqrt{p}$$, 회귀문제에서는 $$p/3$$이다(랜덤 포레스트를 만든 Leo Breiman과 Adele Cutler가 추천하는 값이다). 랜덤 포레스트가 m에 민감하지 않기 때문에 굳이 m에 대한 미세조정(fine tuning)을 할 필요는 없다고 한다 [[2]](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-3).

파이썬을 이용한 예제를 통해 이에 대해 살펴보자. scikit learn 라이브러리에서 불러올 수 있는 boston 데이터를 사용하였다.

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
  <img src="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/nfeatures.png?raw=true" title="nfeatures" width="400">
</a>

6일 때가 가장 낮은 값을 보이지만 $p/3$에 가까운 4도 이와 거의 비슷한 결과를 보이고 있다.

## Out of Bag Samples

배깅된 트리는 평균적으로 관측치들의 약 2/3을 이용한다. Out of Bag(이하 OOB) 샘플은 배깅된 트리를 적합하는데 사용되지 않은 나머지 약 1/3의 관측치들이다.

랜덤 포레스트는 배깅을 기반으로 하기 때문에 OOB 샘플을 test set 대신 사용하여 OOB error를 구할 수 있다. OOB error는 test error와 거의 비슷한 값을 준다고 알려져 있다. 이를 Boston 데이터를 통해 살펴보자.

{% highlight python %}
cv_error=[]
cv_mse=[]
N = len(all_boston)
random.seed(0)
idx_fold = np.random.choice(np.arange(5), N, replace=True)
for ntree in range(20, 510, 10):
    temp_mse = []
    temp_error = []
    for fold in range(5):
        if ntree % 10 == 0 : print(fold+1, "th fold ", "ntree=",ntree)
        idx_test = idx_fold==fold
        idx_train = idx_fold!=fold
        X_train = all_boston.iloc[idx_train, 0:13]
        y_train = all_boston.iloc[idx_train, 13]
        X_test = all_boston.iloc[idx_test, 0:13]
        y_test = all_boston.iloc[idx_test, 13]
    
        rf = RandomForestRegressor(n_estimators=ntree, max_features=4, random_state=0, n_jobs=-1, oob_score=True)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        temp_mse.append(np.mean((rf_pred-np.array(y_test).flatten())**2))
        temp_error.append(1-rf.score(X_test,y_test))
    cv_mse.append(np.mean(temp_mse))
    cv_error.append(np.mean(temp_error))
{% endhighlight %}

{% highlight python %}
# Draw plot
ntree = range(20, 510, 10)
plt.style.use('ggplot')
plt.plot(ntree, oob_error, label="OOB error")
plt.plot(ntree, cv_error, label="CV error")
plt.xlabel('Number of trees')
plt.ylabel('Error')
plt.legend()
plt.show()
{% endhighlight %}

<a href="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/oob.png?raw=true" data-lightbox="oob" data-title="oob">
  <img src="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/oob.png?raw=true" title="oob" width="400">
</a>

## Feature importance

변수 중요도는 OOB 데이터를 permuting 해서 계산된다. 각각의 트리에 대해, OOB 데이터에 대한 prediction error(분류인 경우 error rate, 회귀인 경우 MSE)를 계산한다. 그런 다음 같은 작업을 각 변수들의 순서를 섞어서(permutation) 진행한다. 이렇게 계산된 두 개의 prediction error의 차이를 모든 트리에 대해 평균을 낸 후 차이의 표준편차로 표준화를 해서 계산된 값이 변수 중요도이다.

{% highlight python %}
# Feature importance
features = list(all_boston)[0:13]
importances=rf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
{% endhighlight %}

<a href="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/imp.png?raw=true" data-lightbox="imp" data-title="imp">
  <img src="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/imp.png?raw=true" title="imp" width="400">
</a>

## User defined function

[https://github.com/llSourcell/random_forests/blob/master/Random%20Forests%20.ipynb](https://github.com/llSourcell/random_forests/blob/master/Random%20Forests%20.ipynb)

위의 코드를 조금 수정하여 사용자 정의 함수로 랜덤 포레스트를 구현해 보았다. pandas dataframe 형태로 데이터를 넣어도 코드가 실행 되도록 조정하였다. 또한, random seed를 부여하여 출력되는 값을 고정하려고 하였는데 랜덤 함수가 들어가는 부분에는 모두 seed를 붙여였지만 출력값이 계속 달라지는 결과를 얻었다. 이 부분은 추후 수정이 필요해 보인다.


{% highlight python %}
# Load libraries
import numpy as np
import random
from math import sqrt


# Build a decision tree
def build_tree(dataset, max_depth, min_size, n_features):
    # Building the tree involves creating the root node and 
	root = get_split(dataset, n_features, random_state)
    # calling the split() function that then calls itself recursively to build out the whole tree.
	split(root, max_depth, min_size, n_features, 1)
	return root

# 각 노드에서 best split을 찾는데 사용하는 함수
def get_split(dataset, n_features, random_state):
    class_values = list(set([row[-1] for row in dataset]))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    # 각 split 마다 사용할 변수를 전체 변수 중 n_features 만큼만 랜덤하게 선택 
    random.seed(random_state)
    features = list(np.random.choice(range(len(dataset[0])-1), n_features, replace=False))
    
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Left child node, Right child node로 관측값들 분류
def test_split(index, value, dataset):
	left, right = [], []
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# gini index 계산
def gini_index(groups, class_values):
	gini = 0.0
	for class_value in class_values:
		for group in groups:
			size = len(group)
			if size == 0:
				continue
			proportion = [row[-1] for row in group].count(class_value) / float(size)
			gini += (proportion * (1.0 - proportion))
	return gini

def to_terminal(group):
    #select a class value for a group of rows. 
	outcomes = [row[-1] for row in group]
    #returns the most common output value in a list of rows.
	return max(set(outcomes), key=outcomes.count)
 
# 새로운 child node를 만들거나 더 이상 split 할 수 없으면 terminal node 생성
def split(node, max_depth, min_size, n_features, depth):
    
	left, right = node['groups']
	del(node['groups']) # 더 이상 필요 없어서 제거
    
    # Left, Right child node에 인스턴스들이 있는지 확인하고 없으면 terminal node 생성
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
    
    # Maximum depth에 도달하면 terminal node 생성 
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
    
    # We then process the left child, creating a terminal node if the group of rows is too small, 
    # otherwise creating and adding the left node in a depth first fashion until the bottom of 
    # the tree is reached on this branch.

    # process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, n_features, random_state)
		split(node['left'], max_depth, min_size, n_features, depth+1)
	
    # process right child
    # The right side is then processed in the same manner, 
    # as we rise back up the constructed tree to the root.
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, n_features, random_state)
		split(node['right'], max_depth, min_size, n_features, depth+1)
 
    
# Make a prediction with a decision tree
def predict(node, row):
    # Making predictions with a decision tree involves navigating the  tree with the specifically provided row of data.
    # Again, we can implement this using a recursive function, where the same prediction routine is 
    # called again with the left or the right child nodes, depending on how the split affects the provided data.
    # We must check if a child node is either a terminal value to be returned as the prediction
    # , or if it is a dictionary node containing another level of the tree to be considered.
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
 
# Bootstrap sample index 추출
def bootstrap(dataset, random_state):
    random.seed(random_state)
    N = len(dataset)
    idx = list(np.random.choice(np.arange(N), np.int(N), replace=False))
    bootstrap_sample = list()
    for i in idx:
        bootstrap_sample.append(dataset[i])
    return(bootstrap_sample)

# Make a prediction with a list of bagged trees responsible for making a prediction with each decision tree and 
# combining the predictions into a single return value. 
# This is achieved by selecting the most common prediction from the list of predictions made by the bagged trees.
def bagging_predict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)
 

# Random Forest main function
# responsible for creating the samples of the training dataset, training a decision tree on each,
# then making predictions on the test dataset using the list of bagged trees.
def randomForest(train, test, max_depth, min_size, n_trees, n_features, random_state):
	trees = list()
	for i in range(n_trees):
		bootstrap_sample = bootstrap(train, random_state)
		tree = build_tree(bootstrap_sample, max_depth, min_size, n_features)
		trees.append(tree)
	predictions = [bagging_predict(trees, row) for row in test]
	return(predictions)
{% endhighlight %}

이번에는 classification 문제에 랜덤포레스트를 사용하기 위해 scikit learn에서 불러올 수 있는 breast cancer 데이터를 사용하였다.

{% highlight python %}
import pandas as pd
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()
all_data = pd.concat([pd.DataFrame(bc.data, columns=bc.feature_names), pd.DataFrame(bc.target, columns=["target"])], axis=1)

# split test & train data
N = len(all_data)
ratio = 0.75
random.seed(0)
idx_train = list(np.random.choice(np.arange(N), np.int(ratio * N), replace=False))
idx_test = list(set(np.arange(N)).difference(idx_train))  

train = all_data.iloc[idx_train,:]
test = all_data.iloc[idx_test,:]

# pandas dataframe to list
def transform_data(pd_dataframe):
    dataset = list()
    for i in range(len(pd_dataframe)):
        dataset.append(pd_dataframe.iloc[i,:].tolist())
        dataset[i][-1] = int(dataset[i][-1] ) 
    return(dataset)

traindata = transform_data(train)
testdata = transform_data(test)
  
# Hyperparameter 지정
random_state=0
max_depth = 10
min_size = 1
n_features = int(sqrt(len(traindata[0])-1))
n_trees = 20

random.seed(0)
rf_pred = randomForest(traindata, testdata, max_depth, min_size, n_trees, n_features, random_state)
y_test = [ row[-1] for row in testdata ]
accuracy = np.mean(np.array(y_test) == np.array(rf_pred))
accuracy # 0.9440559440559441
# 왜 seed 고정이 안될까...
{% endhighlight %}

scikit learn의 randomForestClassifier를 이용한 결과와 비교해보자.

{% highlight python %}
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()
bc.DESCR

all_data = pd.concat([pd.DataFrame(bc.data, columns=bc.feature_names), pd.DataFrame(bc.target, columns=["target"])], axis=1)

N = len(all_data)
ratio = 0.75
random.seed(0)
idx_train = np.random.choice(np.arange(N), np.int(ratio * N), replace=False)
idx_test = list(set(np.arange(N)).difference(idx_train))

X_train = all_data.iloc[idx_train, 0:30]
y_train = all_data.iloc[idx_train, 30]
X_test = all_data.iloc[idx_test, 0:30]
y_test = all_data.iloc[idx_test, 30]

random.seed(0)
rf = RandomForestClassifier(n_estimators=20, oob_score=True, 
                            max_depth=10, max_features=int(round(np.sqrt(X_train.shape[1]))),
                            random_state=0, n_jobs=-1)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
accuracy = np.mean(y_test == rf_pred)
print(f'Mean accuracy score: {accuracy:.3}')
{% endhighlight %}

Mean accuracy score: 0.965

# Rotation Forest

## Intro

로테이션 포레스트는 특징 추출(feature extraction)에 기반한 앙상블 기법이다. base learner가 사용할 훈련 데이터(training data)를 만들기 위해, 예측변수들을 K개의 부분집합(subset)으로 랜덤하게 분할한 후 주성분 분석(PCA; Principal Component Analysis)를 각각의 부분집합에 적용한다. 데이터의 변동성(variablility) 정보를 보존하기 위해 모든 주성분(Principal Components)들이 사용된다. 결과적으로, base learner가 사용할 새로운 예측변수들을 만들기 위해 K개의 축 회전이 발생하는 것이다. 축을 회전시키는 접근은 앙상블 모델의 다양성을 높이기 위한 것이다. 다양성이 높아지는 이유는 축 회전을 통해 새로운 변수들이 추출(feature extraction)되었기 때문이다[3]). 로테이션 포레스트는 사선 방향으로 결정 경계(decision boundary)를 만들 수 있다는 장점이 있지만, 분산 방향이 특정 구조로 나타나는 데이터가 아니라면 성능향상을 기대하기 어렵다는 단점이 있다.

## Algorithm

<a href="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/rotation_forest1.png?raw=true" data-lightbox="rotation_forest1" data-title="rotation_forest1">
  <img src="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/rotation_forest1.png?raw=true" title="rotation_forest1" width="400">
</a>

<a href="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/rotation_forest2.png?raw=true" data-lightbox="rotation_forest2" data-title="rotation_forest2">
  <img src="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/rotation_forest2.png?raw=true" title="rotation_forest2" width="400">
</a>

<a href="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/rotation_forest3.png?raw=true" data-lightbox="rotation_forest3" data-title="rotation_forest3">
  <img src="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/rotation_forest3.png?raw=true" title="rotation_forest3" width="400">
</a>

<a href="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/rotation_forest4.png?raw=true" data-lightbox="rotation_forest4" data-title="rotation_forest4">
  <img src="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/rotation_forest4.png?raw=true" title="rotation_forest4" width="400">
</a>

<a href="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/rotation_forest5.png?raw=true" data-lightbox="rotation_forest5" data-title="rotation_forest5">
  <img src="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/rotation_forest5.png?raw=true" title="rotation_forest5" width="400">
</a>

# Decision Jungle

## Intro

앞에서 살펴본 랜덤 포레스트는 모델을 구축하는데 드는 시간이 짧고, 예측 정확도가 높다는 장점이 있다. 하지만 트리를 전개시키는 과정에서 노드 수가 기하급수적으로 증가하기 때문에 메모리가 많이 필요하다는 단점이 있다. 모바일이나 임베디드 프로세서의 경우 메모리 리소스가 제한적이기 때문에 트리의 깊이를 제한 할 수 밖에 없고, 이에 따라 모델의 정확도도 낮아지게 된다. 디시전 정글은 다범주(multi-class) 분류 문제를 풀기 위해 만들어진 [DAG(Directed Acyclic Graphs)](https://en.wikipedia.org/wiki/Directed_acyclic_graph) 모델의 앙상블이다. 마이크로소프트 리서치에서 개발한 알고리즘이고, 마이크로소프트의 머신러닝 플랫폼인 [Azure](https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/multiclass-decision-jungle)에 적용되어 있다. 전통적인 트리 모델과는 달리 디시전 정글은 parent node들이 모든 child node들과 연결되는 것을 허용한다. 좀 더 정확히 말하면, 두 개 이상의 child node 사이의 병합을 허용하는 것이다. 디시전 정글은 랜덤 포레스트보다 모델 구축에 걸리는 시간은 증가하지만 메모리 사용은 획기적으로 감소하고, 예측 성능도 확보할 수 있다[[4]](http://www.nowozin.net/sebastian/papers/shotton2013jungles.pdf).

## Algorithm

디시전 정글의 original paper([[4]](http://www.nowozin.net/sebastian/papers/shotton2013jungles.pdf))를 보면 아래의 Figure를 통해 알고리즘을 설명하고 있다.

<a href="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/df_fig1.PNG?raw=true" data-lightbox="dj_fig1" data-title="dj_fig1">
  <img src="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/dj_fig1.PNG?raw=true" title="dj_fig1" width="400">
</a>

(b)에서 $$N_p$$는 parent node들의 집합이고, $$N_c$$는 child node들의 집합이다. 여기서 child node의 maximum 값을 지정하여 노드 수가 기하급수적으로 늘어나는 것을 방지한다. $$\theta_i$$는 i번째 parent node가 split에 어떤 변수를 사용했는지, split point가 어디인지를 나타내는 파라미터들이다. $$S_i^L$$은 i번째 parent node의 left child node이고, $$S_i^R$$은 i번째 parent node의 right child node이다.

$$l_i$$와 $$r_i$$는 i번째 parent node에서 어떻게 split이 되었고, i번째 parent node의 left child node와 right child node가 무엇인지가 다 로그로 기록된다는 것을 의미한다. 이 로그 데이터를 바탕으로 최적의 split을 찾는 과정을 반복하는데, 이러한 점 때문에 모델 구축 과정이 랜덤 포레스트보다 더 오래 걸리게 된다.

디시전 정글의 objective function은 다음과 같다:

$$E(\{ \theta \}, \{l_i\}, \{r_i\})=\sum_{j \in N_c}|S_j|H(S_j)$$

여기서 $$\|S_j\|$$는 j번째 child node의 인스턴스 개수이고, $$H(S_j)$$는 child node의 entropy이다. 이제 ojective function을 minimize 시켜야 하는데 이를 direct로 푸는 것은 쉽지 않아서 근사적인 방법을 사용해야 한다.

### Lsearch

Lsearch의 핵심은 feasible solution으로 부터 트리 분할을 시작한다는 것이다. 다시 말해, random한 split point에서 출발한다는 것을 의미한다. 파라미터들을 랜덤하게 할당한 후 `split optimization`과 `branch optimization`을 반복적으로 수행하는 것이 Lsearch이다. split optimization 단계에서는 parent node에서 child node로 가는 화살표는 모두 고정한 상태에서 각각의 parent node의 split point를 바꿔가면서 전체 entropy가 가장 낮게 하는 split 조합을 찾는 것을 목적으로 한다. branch optimization 단계에서는 left child node와 right child node를 어떻게 할당하는 것이 최적인지 찾는다. 하나의 parent node의 최적 left/right child node 할당을 찾는 과정에서 나머지 parent 노드들의 화살표는 고정된다. 그런 다음 split optimization과 branch optimization을 더 이상 전체 entropy의 감소가 없을 때까지 반복한다.


### Cluster Search

Cluster search는 branch optimization 과정이 global하게 진행된다는 점에서 Lsearch와 차이가 있다. 먼저 $$\|2N_p\|$$개의 temporary child node들이 전통적이 트리 기반 방법을 통해 만들어진다. 그 다음에는 temporary node들이 $$M=\|N_c\|$$개의 그룹으로 클러스터링 된다. 다시 말해, node clustering을 통해 child node를 병합하는 것이다.

{% highlight python %}

{% endhighlight %}

# Reference

[1] Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009). The Elments of Statistical Learning. Springer.

[2] Ramon Diaz-Uriarte, Sara Alvarez (2006). Gene Selection and Classification of Microarray Data Using Random Forest

[3] Juan J. Rodrı´guez, Ludmila I. Kuncheva. (2006). Rotation Forest: A New Classifier Ensemble Method

[4] Jamie Shotton et.al. (2013). Decision Jungles:Compact and Rich Models for Classification