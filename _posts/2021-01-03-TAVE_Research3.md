---
layout: post
title: Neural Network III
subtitle: loss function, entropy, neural Network
tags: [TAVE, TAVE Research, Deep Learning]
author: marshmello
comments: True
date: 2021-01-03
lastmod: 2021-01-03 00:55:00
sitemap:
changefreq: daily
priority: 1.0
use_math: true
---

## Loss Function

- 신경망에서는 하나의 지표를 기준으로 최적의 매개변수 값을 탐색한다. 신경망 학습에서 사용하는 지표는 손실 함수(loss function)이다.

### SSE(sum of squares for error)

\begin{align}
E = {1/2} {\sum}\_k{(y_k - t_k)^2}
\end{align}

- $y_k$는 신경망이 추정한 값, $t_k$는 정답 레이블, $k$는 데이터 차원의 수를 나타낸다.

```python
def SSE(y, t):
    return 0.5 * np.sum((y - t) ** 2)
```

```python
# t = 정답 레이블
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

# y = softmax함수의 출력
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])

# y2 = softmax함수의 출력
y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.6, 0.3, 0.0, 0.0])

SSE(y, t)  # 0.09750000000000003
SSE(y2, t) # 0.6425
```

- 정답은 '2'인 배열 안에서 SSE의 기준으로 첫 번째 추정 결과가 오차가 더 작기 때문에 정답에 가깝다고 판단 할 수 있다.

## Entropy

- 엔트로피는 정보를 표현하는데 필요한 최소 평균 자원량이다.
- 자원량은 0 또는 1의 bits로 표현 한다.
- 인코딩시 확률이 낮은 것들은 길게 코딩하고 높은 것들은 짧게 코딩한다.

  ![log.png](/assets/img/posts/TaveResearch/neuralN3/log.png)

- y축은 코딩되는 길이, x축은 확률이다.
- 아래 식은 확률에 따른 길이를 나타낸다.

\begin{align}
-\log_2{P_i}
\end{align}

- 자원량의 최소 기댓값이 최소 평균 길이다.

\begin{align}
E = {\sum}\_i {p_i (-\log_2 P_i)}
\end{align}

## CEE(cross entropy error)

### Cross Entropy

- 교차 엔트로피는 추정한 기댓값이다

\begin{align}
E = {\sum}\_i{p_i(-log_2 Q_i)}
\end{align}

```python
def CEE(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
```

```python
# t = 정답 레이블
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

# y = softmax함수의 출력
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])

# y2 = softmax함수의 출력
y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.6, 0.3, 0.0, 0.0])

CEE(y, t)  # 0.510825457099338
CEE(y2, t) # 2.302584092994546
```

- 오차 제곱합의 예시와 같이 정답은 '2'이다.
- delta는 np.log를 계산할 대 0이 입력되면 마이너스 무한대(-inf)가 되는 것을 방지하기 위해 아주 작은 수를 사용하였다.
- 오차값이 더 작은 첫 번째 추정이 정답이다.

## KL Divergence(Kullback–Leibler divergence)

\begin{align}
KL(p||q) = -{\sum}\_i {p_i \log({q_i / p_i})}
\end{align}

- KL divergence는 p와 q의 cross-entropy에서 p의 entropy를 뺀 값이다.
- cross entropy를 minimize 하는 것은 KL divergence를 minimize하는것과 같다. (p의 entropy는 고정된 값이기 때문)
  > [초보를 위한 정보이론 안내서 - KL divergence 쉽게 보기](https://hyunw.kim/blog/2017/10/27/KL_divergence.html)

## 학습 알고리즘 구현

- 2층 신경망을 구현해 본다.

1. 미니 배치

   훈련 데이터 중 일부를 무작위로 가져온다. 가져온 미니배치의 손실 함수 값을 줄이는것이 목표이다.

2. 기울기 산출

   미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구한다. 기울기는 손실 함수의 값을 가장 작게 하는 방향을 제시한다.

3. 매개변수 갱신

   가중치 매개변수를 기울기 방향으로 조금씩 갱신한다.

4. 1~3을 반복한다.

## two_layer_net.py

```python
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
```

- \_\_init\_\_에서는 입력층과 은닉층의 뉴런 수, 출력층의 뉴런 수 예측을 인수로 받는다.
- predict함수는 예측을 수행한다. 파라미터 x는 이미지 데이터이다.
- accuracy함수는 정확도를 구한다.
- gradient함수는 가중치 매개변수의 기울기를 구한다. 오차 역전파(backward)를 사용하여 수치 미분을 사용하는것 보다 시간이 단축된다.

### miniBatch_train_nNet.py

```python
import sys, os

# 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100  # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
```

![loss_acc.png](/assets/img/posts/TaveResearch/neuralN3/loss_acc.png)

- 미니배치 크기를 100으로 하여 한번의 iteration을 수행할 때마다 임의로 100개의 데이터를 추린다.
- 100개의 미니배치를 대상으로 확률적 경사 하강법을 수행하여 매개변수를 갱신한다.
- 갱신할 때마다 훈련 데이터에 대한 손실 함수를 계산한다.
- 왼쪽 그래프는 각 iteration마다의 loss를 나타낸 것이고 오른쪽 그래프는 epch마다 훈련 데이터와 시험 데이터에 대한 정확도를 나타냈다.
- 훈련 데이터와 시험 데이터가 차이가 거의 없는것을 볼 수 있다.

###### References

> <sub>코드: [밑바닥부터 시작하는 딥러닝](https://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)</sub>  
> <sub>YouTube: [혁펜하임](https://www.youtube.com/channel/UCcbPAIfCa4q0x7x8yFXmBag)</sub>
