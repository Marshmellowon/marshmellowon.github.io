---
layout: post
title: "Neural Network V"
subtitle: optimizer, batch normalization, hyperparameter
tags: [TAVE, TAVE Research, Deep Learning]
author: marshmello
comments: True
date: 2021-01-13
lastmod: 2021-01-14 01:20:00
sitemap:
  changefreq: daily
  priority: 1.0
use_math: true
---

## Update Parameter

### Optimization

- 신경망 학습의 목적은 손실 함수의 값을 가능한 낮추는 매개변수를 찾는 것이다.

### SGD(확률적 경사 하강법)

\begin{align}
W \leftarrow W-\eta ({\partial L / \partial W})
\end{align}

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params, grads:
            params[key] -= self.lr * grads[key]
```

- $\eta$는 learning rate이고, 실제로는 0.01이나 0.001과 같은 값을 정해서 사용한다.
- W는 가중치 매개변수이고, $\partial L / \partial W$는 가중치에 대한 손실함수(LOSS)의 기울기다.
- params와 grads는 딕셔너리 변수이다.
- params['W'], grads['W']와 같이 각각의 가중치 매개변수와 기울기를 저장하고 있다.

### Momentum

\begin{align}
v \leftarrow \alpha v - \eta ({\partial L / \partial W})
\end{align}

\begin{align}
W \leftarrow W+v
\end{align}

```python
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
```

- 초기화에서 v는 값이 없다.
- momentum은 변수 v가 추가되는데 v는 이전 학습결과도 반영한다.
- $\alpha$를 0.9로 설정한다면, 이전 학습결과를 0.9를 반영하고 현재 batch는 0.1을 반영하여 가중치를 업데이트 한다.

### AdaGrad

\begin{align}
h \leftarrow h+{\partial L / \partial W} \odot {\partial L / \partial W}
\end{align}

\begin{align}
W \leftarrow W-\eta ({1 / {\sqrt {h}}})\*({\partial L / \partial W})
\end{align}

```python
class AdaGrad:

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```

- h는 기존 기울기 값을 제곱해서 더해준다.
- 매개변수를 갱신할 때 $1 / \sqrt h$을 곱해 학습률을 조정한다.
- 학습률 감소가 매개변수의 원소마다 다르게 적용된다.
- 학습을 진행할 수록 갱신 강도가 약해지며, 무한히 학습하면 갱신량이 0이되어 갱신이 이루어지지 않는다.
- 이러한 갱신 문제는 RMSProp에서 해결한다.
- 코드에서는 1e-7을 더해주는데 0으로 나누는 일이 없도록 해준다.

### RMSProp

\begin{align}
E\[{\partial {\_w}}^{2} D]\_k = \gamma E\[{\partial {\_w}}^{2} D]\_{k-1} +(1-\gamma)({\partial {\_w}}^{2} D)\_k
\end{align}

\begin{align}
w\_{k+1} = w_k - ({\eta / \sqrt{E\[{\partial {\_w}}^{2} D]\_{k+\epsilon}}}) \* \partial_wD
\end{align}

```python
class RMSprop:

    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```

- 이 코드에서는 decay rate를 0.9로 설정했다.
- Adagrad의 learning rate가 0으로 수렴하는 것을 개선하기 위해 기울기를 단순 누적하지 않고 지수 가중 이동 평균을 사용하여 최신 기울기가 더 크게 반영되도록 했다.

### Adam

\begin{align}
{m_t} = {\beta_1} {m}\_{t-1}+(1-{\beta_1})g_t
\end{align}

\begin{align}
v_t = {\beta_2}v\_{t-1}+(1-\beta_2){g{\_t}}^2
\end{align}

\begin{align}
\hat{m}\_t = {m_t / 1-{\beta{\_1}}^t}
\end{align}

\begin{align}
\hat{v} = {v_t / 1-{\beta{\_2}}^t}
\end{align}

\begin{align}
\theta\_{t+1} = \theta_t-({\eta / \sqrt{\hat{v_t}+\epsilon}}) \hat{m}\_t
\end{align}

```python
class Adam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
```

- Adam optimizer는 RMSProp과 Momentum을 합친 알고리즘이다.
- 1차 Momentum m과 2차 Momentum v를 이용하여 최적화를 진행한다.

## Batch Normalization

### Gradient vanishing, Exploding problem

- 신경망 학습시 gradient(변화량)이 매우 작아지거나 커지면 신경망을 제대로 학습시키지 못한다.
- 이런 문제를 해결하기 위한 트릭은 activation function 바꾸기(ReLU), 가중치 초기화하기, learning rate를 작게 하기가 있다.
- 학습 과정을 전체적으로 안정적이게 하여 학습 속도를 높여주는 Batch Normalization을 사용할 수 있다.

  ### Batch Normalization

  ![/assets/img/posts/TaveResearch/neuralN5/batchN.png](/assets/img/posts/TaveResearch/neuralN5/batchN.png)

- 배치 정규화는 신경망 안에 포함되어 학습시 평균과 분산을 조정하는 역할을 한다.
- 레이어 마다 정규화 하는 레이어를 둬서 변형된 분포가 나오지 않게 한다.
- 미니 배치의 평균과 분산을 이용하여 정규화 한 후 scaling과 shift를 진행한다.($\gamma$ 와 $\beta$값을 이용)

![/assets/img/posts/TaveResearch/neuralN5/covariate.png](/assets/img/posts/TaveResearch/neuralN5/covariate.png)

\begin{align}
u_i = {z_i -\mu_B / \sqrt {\sigma{\_B}^2 +\epsilon}}
\end{align}

\begin{align}
\hat {z}\_i = \gamma u_i +\beta
\end{align}

- 미니 배치의 평균은 $\mu_B = {1 / B}{\sum_{i=1}}^B {z_i}$이고, 분산은 $\sigma{\_B}^2 = 1/B {\sum{^B}}\_{i=1} {(z_i - \mu_B)} $이다.
- $u_i$은 정규화 식이고, $\hat{z}_i$는 scaling과 shift이다.

## For right learning

- 기계학습에서는 overfitting이 문제가 되는일이 많다.
- 신경망이 훈련 데이터에만 지나치게 적으오디어 그 외의 데이터에는 제대로 대응하지 못하는 상태를 overfitting이라고 한다.

### Overfitting

- overfitting이 일어나는 경우
  - 매개변수가 많고 표현력이 높은 모델
  - 훈련 데이터가 적은 경우

```python
# coding: utf-8
import os
import sys

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# weight decay（가중치 감쇠） 설정 =======================
# weight_decay_lambda = 0 # weight decay를 사용하지 않을 경우
weight_decay_lambda = 0.1
# ====================================================

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        )
optimizer = SGD(lr=0.01)  # 학습률이 0.01인 SGD로 매개변수 갱신

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
```

- 60,000개인 MNIST 데이터셋의 훈련 데이터중 300개만 사용하고 7층 네트워크를 사용하였다.
- 각 층의 뉴런 갯수는 100개, activation function은 ReLU를 사용하였다.

  ![/assets/img/posts/TaveResearch/neuralN5/overfit_weight_decay.png](/assets/img/posts/TaveResearch/neuralN5/overfit_weight_decay.png)

- 훈련 데이터를 사용한 것과 시험 데이터를 사용한 것의 정확도 차이가 크게 나는것을 볼 수 있다.
- epoch마다 모든 훈련 데이터와 모든 시험 데이터에서 정확도를 산출한다.
- 훈련 데이터는 100 epoch 부분부터 100의 정확도를 보인다.

### Weight decay

- overfitting을 억제하기 위해 weight decay(가중치 감소)를 사용한다.
- 학습 과정 중에서 큰 가중치에 대해서는 그에 따른 큰 페널티를 부여하여 overfitting을 억제한다.
- 가중치 감소는 모든 가중치 각각의 손실 함수에 ${1 / 2}\lambda W$를 더해준다.
- 가중치의 기울기를 구하는 계산에서는 Backpropagation에 따른 결과에 정규화 항을 미분한 $\lambda W$를 더해준다.
- 가중치 감소를 추가해 준다.

  ```python
  def loss(self, x, t):
          y = self.predict(x)

          weight_decay = 0
          for idx in range(1, self.hidden_layer_num + 2):
              W = self.params['W' + str(idx)]
              weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

          return self.last_layer.forward(y, t) + weight_decay
  ```

  ```python
  def gradient(self, x, t):
          # forward

          # backward

          # 결과 저장
          grads = {}
          for idx in range(1, self.hidden_layer_num+2):
              grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
              grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

          return grads
  ```

  ```python
  weight_decay_lambda = 0.1
  network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                          weight_decay_lambda=weight_decay_lambda)
  ```

  ![/assets/img/posts/TaveResearch/neuralN5/overfit_weight_decay%201.png](/assets/img/posts/TaveResearch/neuralN5/overfit_weight_decay%201.png)

  - overfitting이 억제된 것을 볼 수 있고, 훈련 데이터에 대한 정확도가 떨어진 것을 볼 수 있다.

### Dropout

- 뉴런을 임의로 삭제하면서 학습하는 방법이다.
- 은닉층의 뉴런을 무작위로 골라 삭제한다.
- 훈련 시에는 삭제할 뉴런을 무작위로 선택하고, 시험 때는 모든 뉴런에 신호를 전달한다.

![/assets/img/posts/TaveResearch/neuralN5/dropout.png](/assets/img/posts/TaveResearch/neuralN5/dropout.png)

```python
class Dropout:

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
```

- self.mask는 x와 형상이 같은 배열을 무작위로 생성하고 그 값이 0.5(dropout_ratio)보다 큰 원소만 True로 설정한다.

  ![/assets/img/posts/TaveResearch/neuralN5/dropout2.png](/assets/img/posts/TaveResearch/neuralN5/dropout2.png)

## Find the value of the appropriate hyperparameter

- 하이퍼파라미터는 각 층의 뉴런 수 , 배치 크기, 매개변수 갱신 시 학습률, 가중치 감소 등이다.

### Validation data

- 하이퍼파라미터를 조정할 때 사용하는 데이터를 validation data(검증 데이터)라고 한다.
- MNIST 데이터셋에서 검증 데이터를 얻는 방법은 훈련 데이터 중 20% 정도를 검증 데이터로 분리하는 것이다.

```python
from common.util import shuffle_dataset

(x_train, t_train), (x_test, t_test) = load_mnist()

# 훈련 데이터를 섞는다
x_train, t_train = shuffle_dataset(x_train, t_train)

# 20%를 검증 데이터로 분할
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)

x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]
```

### Hyperparameter optimization

- 0단계

  하이퍼파라미터 값의 범위를 설정한다

- 1단계

  설정된 범위에서 하이퍼파라미터의 값을 무작위로 추출한다

- 2단계

  1단계에서 샘플링한 하이퍼파라미터 값을 사용하여 학습하고, 검증 데이터로 정확도를 평가한다(epoch은 작게 설정한다.)

- 3단계

  1단계와 2단계를 특정 횟수 반복하여 그 정확도의 결과를 보고 하이퍼파라미터의 범위를 좁힌다

```python
# 탐색한 하이퍼파라미터의 범위 지정
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)
```

- 전체 코드는 [hyperparameter_optimization.py](https://github.com/Marshmellowon/DeepLearning_from_Scratch_TAVEr/blob/master/lec7_lec8/hyperparameter_optimization.py)에 있다.

> 이 글에 나왔던 common 폴더는 \[[common](https://github.com/Marshmellowon/DeepLearning_from_Scratch_TAVEr/tree/master/common)] 여기에 있다.

###### References

> <sub>강의: [CMU Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/F20/index.html)</sub>  
> <sub>코드: [밑바닥부터 시작하는 딥러닝](https://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)</sub>
