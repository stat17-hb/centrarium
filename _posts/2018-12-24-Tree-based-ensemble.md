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

6일 때가 가장 낮은 값을 보이지만 $$p/3$$에 가까운 4도 이와 거의 비슷한 결과를 보이고 있다.

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

## Example

[https://drive.google.com/file/d/0B0tdfxikEBvtVnpOdXNKQUd2S2M/view](https://drive.google.com/file/d/0B0tdfxikEBvtVnpOdXNKQUd2S2M/view)

위의 코드를 기반으로 디시전 정글 예제 코드를 만들었다. itemfreq 대신 np.unique를 사용하여 major class 개수를 셋고, pd.concat을 할 때 row index가 맞지 않으면 새로운 행들이 추가되어 NA 값이 채워지는 것을 막기 위해 row index를 맞춰주었다. 랜덤 포레스트에서와 마찬가지로 seed를 고정하여 같은 결과를 reproduce해 보려고 하였지만 이 코드에서도 seed가 고정되지 않았다. 이 부분은 추후에 수정이 필요해 보인다. 예제 데이터로는 랜덤 포레스트에서와 같은 breast cancer 데이터를 사용하였다.

{% highlight python %}
import math
import numpy as np
import pandas as pd
from scipy.stats import entropy
import random

# Load data
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()
all_data = pd.concat([pd.DataFrame(bc.data, columns=bc.feature_names), pd.DataFrame(bc.target, columns=["Y"])], axis=1)

# split test & train data
N = len(all_data)
ratio = 0.75
random.seed(0)
idx_train = list(np.random.choice(np.arange(N), np.int(ratio * N), replace=False))
idx_test = list(set(np.arange(N)).difference(idx_train))  

train = all_data.iloc[idx_train,:]
test = all_data.iloc[idx_test,:]


# 파이썬에 which 함수가 없어 정의
which = lambda lst:list(np.where(lst)[0])

# 레이블을 담고 있는 인덱스와 Feature를 담고 있는 인덱스 지정
# 레이블 인덱스의 이름은 반드시 Y 가 되어야 함 (엔트로피 계산시에도 사용)
idx_label = which(train.columns==u"Y")
idx_feature = which(train.columns!=u"Y")

limit_w=4
limit_d=10 # max_depth
op_select = "sqrt"
prob = 0.8

#### Main fucntion ####

def model_dj (data,               # traning dataset
              idx_feature,        # index of the features 
              idx_label,          # index of the label
              limit_w,            # limit of width (2^w)
              limit_d,            # limit of depth
              op_select = "full", # one of "full, sqrt, prob, log2"
              prob = 0.8):          # if op_select is prob, ratio of choice

    ## dataframe of attributes of a tree
    tree = {'dim'   : np.repeat(-1,cum_numNodes(limit_d, limit_w)),
        'theta' : np.repeat(0,cum_numNodes(limit_d, limit_w)),
        'l'     : np.repeat(0,cum_numNodes(limit_d, limit_w)),
        'r'     : np.repeat(0,cum_numNodes(limit_d, limit_w)),
        'class_': np.repeat(' ',cum_numNodes(limit_d, limit_w))       
        }
    
    tree = pd.DataFrame(tree)

    ## add columns
    
    Np = pd.Series(np.repeat(1, len(data)))
    Nc = pd.Series(np.repeat(1, len(data)))
    
    
    temp = {'Np' : pd.Series(np.repeat(1, len(data))),
            'Nc' : pd.Series(np.repeat(1, len(data))),
            'idx': pd.Series(range(1,len(data)+1))}
    
    temp = pd.DataFrame(temp)
    temp.index = data.index # index를 맞춰줘야 제대로 concat 이 됨!!!
    data = pd.concat([data,temp],axis=1) # len(data)=426, len(temp)=426인데 len(pd.concat([data,temp],axis=1))=534??
    
    ## initialize tree edges
    tree = init_edges(tree)
    # 트리는 현재 레벨까지의 총 노드 수 만큼 딤, 쎄타, 엘, 알, 클래스를 초기화하였다.

    ## decision jungle algorithm
    for depth in range(1, limit_d+1):
        # j는 1부터 제한 뎁스까지 증가
        print("****************","depth =", depth, "****************")
        
        Np = np.unique((data.loc[:,'Nc']))      
        print("Np:", Np)
    
        terminal_flag = True
        dims = select_feature(f = idx_feature, option = op_select, prob = prob) # dims: 선택된 변수의 열 인덱스       

        for i in Np:
            print("-----------------","Np Node " + str(i), "-----------------")
            subdata = data[data.loc[:,'Nc'] == i]
            idx_subdata = subdata.idx
            tmp = find_majorClass(subdata, idx_label) # index 1 is out of bounds for axis 0 with size 0
            tree.loc[i-1,'class_'] = tmp
            
            if (len(subdata) == 0):
                print("Sub Data 가 0인 경우")
                continue  # 'cause parent node is pure, one of children has all data
            
            
            if (H(subdata, idx_label) == 0):
                print ("H 값이 0인 경우")
                data.loc[data.idx.isin(idx_subdata),"Np"] = data.loc[data.idx.isin(idx_subdata),"Nc"]
                data.loc[data.idx.isin(idx_subdata),"Nc"] = find_Nc(i)[0]
                tree.loc[i-1,"dim"] = 1
                tree.loc[i-1,"theta"] = np.inf
                continue
   
            split_info = split_data(subdata, dims, idx_label)

            data.loc[data.idx.isin(split_info['l']), "Np"] = data.loc[data.idx.isin(split_info['l']), "Nc"]
            data.loc[data.idx.isin(split_info['l']), "Nc"] = find_Nc(i)[0]
            data.loc[data.idx.isin(split_info['r']), "Np"] = data.loc[data.idx.isin(split_info['r']), "Nc"]
            data.loc[data.idx.isin(split_info['r']), "Nc"] = find_Nc(i)[1]
                        
            # save split info.
            tree.loc[i-1,"dim"] = split_info['d']
            tree.loc[i-1,"theta"] = split_info['theta']
            terminal_flag = False
    
            # for debug
            print("set threshold : Np is "+ str(i) + " / level is " + str(depth))           
            idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
            print(cal_totalEnt(data, idx_label, idx_node = idx_node))
            #print(tree)
    
        print("set threshold : level is " + str(depth))
        #print(cal_totalEnt(data, idx_label, idx_node = idx_node))
        
        # terminal condition (if child nodes become pure, go out of loop)
        if terminal_flag == True:
            break 
        
        print(is_limitWidth(depth))
        # decision jungle logic below 
        
        if is_limitWidth(depth) == False:
            continue
            
        # 2) update best_deminsion & best_theta
    
        for i in Np:
            subdata = data.loc[data.Np == i, :]

            if len(subdata) == 0:
                continue          # if parent node is empty, go to next parent node
            #if (H(subdata, idx_label) == 0):
                #continue # if parent node is pure, we don't need to adjust threshold
                
            split_info = split_data(data, dims, idx_label, i)

            # update Nc and split info.
            data.loc[data.idx.isin(split_info['l']), "Nc"] = find_Nc(i)[0]
            data.loc[data.idx.isin(split_info['r']), "Nc"] = find_Nc(i)[1]
            tree.loc[i-1,"dim"] = split_info['d']
            tree.loc[i-1,"theta"] = split_info['theta']
            # for debug
            #print ("update threshold : Np is " + str(i) + " / level is " str(j))
            #idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
            #print(cal_totalEnt(data, idx_label, idx_node = idx_node))
            #print(tree)
            
        print("update threshold : level is " + str(depth))
        #print(cal_totalEnt(data, idx_label, idx_node = idx_node))
          

        # 3) update edges
        # left
        for i in Np:
            min_ent = np.inf
            best_edge = 0
            Nc = conv_lTon(depth+1)             # index of child nodes in curent level(parent nodes)
            
            if (len(subdata) == 0):
                continue          # if parent node is empty, go to next parent node
                
            #if (H(subdata, idx_label) == 0) next # if parent node is pure, we don't need to adjust threshold
            
            subdata     = data.loc[(data.Np == i) & (data.Nc == find_Nc(i)[0]),:] # extract data of a left edge Np
            idx_subdata = subdata.idx
            
            # update edge
            for k in Nc:
                if (k == find_Nc(i)[1]):
                    continue       # if current child nodes(right)
                data.loc[data.idx.isin(idx_subdata), "Nc"] = k
                idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
                ent = cal_totalEnt(data, idx_label, idx_node = idx_node)
                
                if(min_ent > ent):
                    min_ent = ent
                    best_edge = k
        
            data.loc[data.idx.isin(idx_subdata),"Nc"] = best_edge
            tree.loc[i-1,"l"] = best_edge

            # for debug
            print("update left edge : Np is " + str(i) + " / level is " + str(depth))
            #idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
            #print(cal_totalEnt(data, idx_label, idx_node = idx_node))
            #print(tree)
            
            
        print("update left edge : level is " +str(depth))
        #idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
        #print(cal_totalEnt(data, idx_label, idx_node = idx_node))
        
        # right
        for i in Np:
            min_ent = np.inf
            best_edge = 0           
            Nc = conv_lTon(depth+1)             # index of child nodes in curent level(parent nodes)
            
            if (len(subdata) == 0):
                continue          # if parent node is empty, go to next parent node
                
            #if (H(subdata, idx_label) == 0) next # if parent node is pure, we don't need to adjust threshold
                        
            subdata     = data.loc[(data.Np == i) & (data.Nc == find_Nc(i)[1]),:] # extract data of a left edge Np
            idx_subdata = subdata.idx

            # update edge
            
            for k in Nc:
                if (tree.loc[i-1,"l"] == k):
                    continue       # if current child nodes(right)
                data.loc[data.idx.isin(idx_subdata), "Nc"] = k
                idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
                ent = cal_totalEnt(data, idx_label, idx_node = idx_node)
                
                if(min_ent > ent):
                    min_ent = ent
                    best_edge = k                    
                    
            data.loc[data.idx.isin(idx_subdata),"Nc"] = best_edge
            if tree.loc[i-1,"l"] > best_edge:
                tree.loc[i-1,"r"] = tree.loc[i-1,"l"]
                tree.loc[i-1,"l"] = best_edge
            else:
                tree.loc[i-1,"r"] = best_edge
                    
            # for debug
            print("update right edge : Np is " + str(i) + " / level is " + str(depth))
            #idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
            #print(cal_totalEnt(data, idx_label, idx_node = idx_node))
            #print(tree)
            
        print("update right edge : level is " +str(depth))
        #idx_node = [z for z, x in enumerate((data.columns == "Nc")) if x][0]
        #print(cal_totalEnt(data, idx_label, idx_node = idx_node))
        
    return tree

#===========================
# S : d 레벨에서 limit_w 라는 제한이 있을 때 사용할 수 있는 노드 수
#===========================
def S (d, D = None):
    if D == None:
        D = limit_w     
    return min(pow(2,d), pow(2,D-1))

#===========================
# is_limitWidth :
# d 가 Width의 제한에 걸렸는지 여부를 리턴
# True / False
#===========================
def is_limitWidth (d, D = None):
    # limit_w - User Parameter
    if D == None:
        D = limit_w
    return ( d > (D-1) )
  
#===========================
# cum_numNodes(level) : 
# 지금까지의 레벨까지 사용한 총 노드 수
#===========================
def cum_numNodes (level, w = None):
    num_nodes = 0
    if w == None:
        w = limit_w
    if (level < w):
        for i in range(level):
            num_nodes += pow(2,i)
    else:
        for i in range(w):
            num_nodes += pow(2,i)
        num_nodes = num_nodes + (level-w)*S(level)     
    return (num_nodes)

#===========================    
# conv_lTon  :
# return nodes # when level is given
# 현재 레벨의 노드 번호 나열
#===========================    
def conv_lTon (level, w = None):
    if w == None:
        w = limit_w
    if (level == 1):
        return 1
    else:
        return range(cum_numNodes(level-1)+1, cum_numNodes(level)+1)
  
#===========================    
# conv_nTol :
# 노드 번호를 이용해서 현재 노드가 속한 레벨을 구한다.
#===========================    
def conv_nTol (node, w = None):
    if w == None:
        w = limit_w
        
    if ( pow(2,w) > node ):
        # 제한에 걸리지 않는 다면
        j = 1
        while (True):
            if(node < pow(2,j)):
                break
            j = j+1
        return (j)
    else:
        return ( ((node-cum_numNodes(w)-1) / S(w))+w+1)

#===========================    
# init_edges : Edge 들을 초기화
#===========================  
def init_edges(tree, d = None, w = None):
    if d == None:
        d = limit_d
    if w == None:
        w = limit_w
    
    max_nodes = cum_numNodes(d) # d 레벨까지 사용한 총 노드 수 = max 노드
    #for i in range(1,int(max_nodes+1)): # 'float' object cannot be interpreted as an integer
    for i in range(1,int(max_nodes+1)): # op_select = "sqrt"이면 error 발생
        if( cum_numNodes(w-1) >= i ):
            tree.loc[i-1,'l'] = i*2
            tree.loc[i-1,'r'] = 2*i+1
        else:
            if (i-cum_numNodes(w-1))% S(w) == 0:
                # if i is last nodes of each level
                tree.loc[i-1,'l'] = i+1
                tree.loc[i-1,'r'] = i+S(w) # connect first and last node of next level
            else:
                tree.loc[i-1,'l'] = i+S(w)
                tree.loc[i-1,'r'] = i+S(w)+1  # connect next level(child node)
    return (tree)    

#===========================    
# find_Nc: 현재 노드의 l, r 을 알려준다.
#=========================== 
def find_Nc(i, w = None):
    if w == None:
        w = limit_w
    if( cum_numNodes(w-1) >= i ):
        return [i*2, 2*i+1]
    else:
        if ((i-cum_numNodes(w-1))%S(w, w) ==  0):
            return [i+1, i+S(w, w)]
        else:
            return [i+S(w, w), i+S(w, w)+1]

#===========================    
# select_feature : 사용할 feature를 리턴
#=========================== 
def select_feature (f, option = "full", prob = 0):
    # select_feature f는 피처를 나타내는 행 리스트 (y행을 뺀 나머지)
    numX = len(f)
    
    if (option == "full"):
        return (f) # 풀인 경우 그대로 다시 내보냄

    if (option == "sqrt"):
        # 제곱근 만큼 피처를 내보냄
        if (round(math.sqrt(numX), 0) < 2):
            n = 2
        else:
            n = int(round(math.sqrt(numX), 0))           
            return sorted(np.random.choice(f, size=n, replace = False))
        # 복원 추출하지 않고, 제곱근 개의 피처를 리턴
  
    if (option == "prob"):
        # 확률 만큼
        if (round(prob*numX , 0) < 2):
            n = 2
        else:
            n = int(round(prob*numX , 0))
        return (sorted(np.random.choice(f, size=n, replace = False)))
        
    if (option == "log2"):
        # 로그 2 만큼
        if (round(math.log(numX,2), 0) < 2):
            n = 2
        else:
            n = int(round(math.log(numX,2), 0))
        return (sorted(np.random.choice(f, size=n, replace = False)))

#===========================    
# find_majorClass : 범주 중에 더 많은 쪽을 알려줌 ---  에러 발생해서 수정함!!
#===========================     
def find_majorClass(data, idx_label):
    print("[Func] find_majorClass")
    x = data.iloc[:,idx_label]
    item = np.unique(x, return_counts=True)
    majorClass = item[0][np.argmax(item[1])]
    return majorClass
            
#===========================    
# H : 섀넌의 엔트로피 계산
#=========================== 
def H(data, label = None):
    if label == None:
        label = idx_label
    if (len(data) == 0):
        return (0)
    
    return entropy(data["Y"].value_counts().tolist(), qk=None, base=2)


#===========================    
# cal_totalEnt : 현재 레벨 전체의 섀넌의 엔트로피 계산
#=========================== 
def cal_totalEnt(data, idx_label, idx_node):
    ent = 0
    if len(data) == 0:
        return (0)
    #if (is.na(data)) return (0)

    node = np.unique(data.iloc[:, idx_node])

    for i in node:
        subdata = data.loc[ data.iloc[:, idx_node] == i,: ]
        ent = ent + len(subdata) * H(subdata, idx_label[0])

    return (ent)

#===========================    
# split_data : 실제 분류하는 부분
#===========================
def split_data (data, idx_feature, idx_label, parent = 'NA'):
    best_theta = best_feature = 0
    min_entropy = np.inf
    
    if H(data, idx_label) == 0:
        return 'NA'
  
    if parent == 'NA':
        print("Decision Tree Algorithm")
        for i in idx_feature:
            #i = 0
            ## initailize variables 
            #print "[+] Split Feature 가 " + str(i) + "번 째 Feature 일 때"
            data_order = data.iloc[np.argsort(data.iloc[:,i]),:]
            # i 피처에 대한 정렬, order하면 해당 위치에 순서를 알려줌
            # 데이터로 감싸야 원래 형태로 바뀜.
            #data_order는 i 에 대해 정렬된 상태                   
            
            #idx = data_order.loc[:,'idx']
            data_i = data_order.iloc[:, i] # 피처 i 에 대해서 정렬된 데이터
            #print data_i
            #j=0                  
            for j in range(0,len(data)-1):
                # skip if ith and (i+1)th data is same or one of exceptRow j = 1
                if (data_i.iloc[j] == data_i.iloc[j+1]):
                    continue
                
                theta = (data_i.iloc[j] + data_i.iloc[j+1]) / 2      
                # Theta는 중간 값을 취한다. 정렬된 값을 반으로 나눔
                #left  = data.loc[ data.iloc[:, idx_feature[i]] <  theta,:]  # Theta보다 작으면 왼쪽
                left  = data.loc[ data.iloc[:, i] <  theta,:]  # Theta보다 작으면 왼쪽

                #right = data.loc[ data.iloc[:, idx_feature[i]] >= theta,:]  # Theta보다 크거나 같으면 오른쪽
                right = data.loc[ data.iloc[:, i] >= theta,:]

                # calcurate entropy
                ent_left  = H(left,  idx_label)  # entropy of left nodes
                ent_right = H(right, idx_label)  # entropy of right nodes
                ent_total = (len(left)*ent_left) + (len(right)*ent_right)
                #전체 엔트로피는 왼쪽의 개체수 곱하기 왼쪽 엔트로피 + 오른쪽 개체수 * 오른쪽 엔트로피
                #print ent_total
                # save better parameters 
                if(min_entropy > ent_total ):
                    min_entropy = ent_total
                    best_theta = theta # 엔트로피가 최소가 되는 Theta (전체 하나 위에꺼 까지 전부다 검색)
                    #best_feature = idx_feature[i] # 어떤 피처인지 찾는다.
                    best_feature = i # 어떤 피처인지 찾는다. 바꾼것.


        # result divided dataset
        left  = data.loc[data.iloc[:, best_feature] <  best_theta,:]  # index out of range
        right = data.loc[data.iloc[:, best_feature] >= best_theta,:] 
        
        result = dict({'d' : best_feature, 'theta': best_theta, 'l':left.idx, 'r': right.idx})
        return result
        
    ## decision jungle logic    
    else:
        print("Decision Jungle Algorithm")
        # extract fixed child nodes(left / right)
        #print "parent : " + str(parent) + "Nc" + str(find_Nc(parent)[0])
        subdata_exRows_l = data.loc[(data.Nc == find_Nc(parent)[0]) & (data.Np != parent),:]
        subdata_exRows_r = data.loc[(data.Nc == find_Nc(parent)[1]) & (data.Np != parent),:]
        
        # extract movable child nodes
        subdata_movable  = data.loc[data.Np == parent,:]
        #print("SUBDATA MOVABLE :" + str(len(subdata_movable)))
        if(len(subdata_movable) == 0):
            best_theta = np.inf
            best_feature = idx_feature[0]

            # result dividing dataset
            left  = subdata_exRows_l
            
            right  = subdata_exRows_r
            
            result = dict({'d' : best_feature, 'theta': best_theta, 'l':left.idx, 'r': subdata_exRows_r})
            return (result)
            
        for i in idx_feature:
            # initailize variables
            #print ("Decision Jungle Feature") + str(i)
            data   = subdata_movable.iloc[ np.argsort(subdata_movable.iloc[:, i]),: ]
            #print "subdata_movable " + str(len(data))
            if(min(data.iloc[:, i]) > 0):
                start = min(data.iloc[:, i])/2
            else:
                start = min(data.iloc[:, i])*2
            if(max(data.iloc[:, i]) > 0):
                end   = min(data.iloc[:, i])*2
            else:
                end   = min(data.iloc[:, i])/2
            
            data_i = []
            data_i.append(start)
            data_i += data.iloc[:, i].tolist()
            data_i.append(end)
            
            for j in range(0, len(data_i)-1):
                
                if data_i[j] == data_i[j+1]:
                    continue
                
                theta = (data_i[j] + data_i[j+1]) / 2
                left  = data.loc[data.iloc[:, i] <  theta,:].append(subdata_exRows_l)
                right  = data.loc[data.iloc[:, i] >=  theta,:].append(subdata_exRows_r)
                
                # calcurate entropy
                
                ent_left  = H(left,  2)  # entropy of left nodes
                ent_right = H(right, 2)  # entropy of right nodes
                ent_total = len(left)*ent_left + len(right)*ent_right
                            
                # save better parameters
                if(min_entropy > ent_total):
                    min_entropy = ent_total
                    best_theta = theta # 엔트로피가 최소가 되는 Theta (전체 하나 위에꺼 까지 전부다 검색)
                    best_feature = i # 어떤 피처인지 찾는다. 수정
        
        # result dividing dataset
        #left  = rbind (subdata_movable.loc[ subdata_movable.iloc[:, best_feature] <  best_theta,: ], subdata_exRows_l)
        left  = subdata_movable.loc[ subdata_movable.iloc[:, best_feature] <  best_theta,: ].append(subdata_exRows_l)
        right = subdata_movable.loc[ subdata_movable.iloc[:, best_feature] >= best_theta,: ]
        
        result = dict({'d' : best_feature, 'theta': best_theta, 'l':left.idx, 'r': right.idx})
        return (result)
		
#===========================    
# predict_dj : 학습한 모델로 실제 predict 하는 부분
#===========================
def predict_dj(tree, data):
    result = list()
    for i in range(0,len(data)):
        #x = data.loc[i,:]
        x = data.iloc[i,:]  # loc => iloc 으로 수정
        node = 1
        
        while (tree.loc[node-1,"l"] <= len(tree) and tree.loc[node-1,"r"] <= len(tree) ):
            dim_l = tree.loc[(tree.loc[node-1,"l"])-1, "dim"]
            #print dim_l
            dim_r = tree.loc[(tree.loc[node-1,"r"])-1, "dim"]
            #print dim_r
            if (dim_l == -1 & dim_r == -1):   # if next node is empty
                break
            
            if ( x.iloc[tree.dim[node-1]] < tree.theta[node-1]):
                if (tree.loc[(tree.l[node-1])-1, "dim"] == -1):
                    break
                else:
                    node = tree.l[node-1] # go a  left node
            else:
                if (tree.loc[(tree.r[node-1]-1), "dim"] == -1):
                    break
                else:
                    node = tree.r[node-1] # go a right node                
                
        if (tree.loc[node-1, "class_"] == " "):
            print("space err")
        if tree.loc[node-1, "class_"] == "NA":
            print("na err")
        if (tree.loc[node-1, "class_"] == 0):
            print("null err")
        
        #print "Node Num "
        #print node-1
        pred = tree.loc[node-1, "class_"]
        result.append(pred)   
    
    return result

# Model fitting
random.seed(0)
dj_fit = model_dj(data = train,
                  idx_feature = idx_feature,
                  idx_label = idx_label,
                  limit_w = 4,
                  limit_d = 10,
                  op_select = "sqrt")

dj_pred = predict_dj(dj_fit, test.iloc[:,0:30])
np.mean(dj_pred == test["Y"])
{% endhighlight %}

<a href="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/dj_results.PNG?raw=true" data-lightbox="dj_results" data-title="dj_results">
  <img src="https://github.com/stat17-hb/stat17-hb.github.io/blob/master/assets/dj_results.PNG?raw=true" title="dj_results" width="400">
</a>


# Reference

[1] Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009). The Elments of Statistical Learning. Springer.

[2] Ramon Diaz-Uriarte, Sara Alvarez (2006). Gene Selection and Classification of Microarray Data Using Random Forest

[3] Juan J. Rodrı´guez, Ludmila I. Kuncheva. (2006). Rotation Forest: A New Classifier Ensemble Method

[4] Jamie Shotton et.al. (2013). Decision Jungles:Compact and Rich Models for Classification