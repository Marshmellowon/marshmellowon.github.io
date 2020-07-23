---
layout: post
title: "Deep Learning for Scratch(3)"
date: 2020-07-22 00:33:25 +0900
categories: [AI, Python]
sitemap:
lastmod: 2020-04-25 17:00:00
changefreq: daily
priority: 1.0
---

<sub>책 정보: [밑바닥부터 시작하는 딥러닝](https://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)</sub>

## Neural Network(신경망)

인공신경망은 사람의 뇌에서 여러개의 뉴런이 연결되면서 복잡한 연산을 수행하는 정보 처리 과정을 모방해서 만든 알고리즘이다.

신경망은 가장 왼쪽에 입력층, 가장 오른쪽에 출력층, 중간을 은닉층이라고 한다.  
총 3개의 층으로 이루어져있다.

![신경망의 예](https://github.com/Marshmellowon/marshmellowon.github.io/blob/master/static/img/_posts/Deep_Learning/no3/Neural_Network.JPG?raw=true)

이전 포스트에서 봤던 퍼셉트론과 크게 다르지 않다. 간단한 퍼셉트론과의 차이는 입력과 출력 이외에 은닉층이 추가된 것 뿐이다.

## 활성화 함수

활성화 함수는 임계값을 경계로 출력이 바뀐다. 이와 같은 함수를 계단 함수라고 한다.

### Sigmoid Function(시그모이드 함수)

![시그모이드 함수식](https://github.com/Marshmellowon/marshmellowon.github.io/blob/master/static/img/_posts/Deep_Learning/no3/sigmoid.JPG?raw=true)

시그모이드 함수는 신경망에서 자주 사용되는 활성화 함수이다.
신경망에서는 활성화 함수로 시그모이드 함수를 이용하여 신호를 변환하고, 다음 뉴런에 전달한다.

### Step Function(계단함수)

기본적인 계단함수를 구현하는 코드이다.

```python
def step_function2(x):
    if x > 0:
        return 1
    else:
        return 0
```

x의 값이 0보다 크면 1을 반환하고 x의 값이 0보다 작거나 같으면 0을 반환한다.

#### - 계단 함수 구현

```python
import numpy as np
import matplotlib.pylab as plt


# 계단 함수
def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
print("step func: ", y)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

```

matplotlib를 사용하여 계단함수 그래프를 그려보았다.

![step function](https://github.com/Marshmellowon/marshmellowon.github.io/blob/master/static/img/_posts/Deep_Learning/no3/step.png?raw=true)

#### - 시그모이드 함수 구현

```python
import numpy as np
import matplotlib.pylab as plt

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

print(sigmoid(x))

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

```

![sigmoid function](https://github.com/Marshmellowon/marshmellowon.github.io/blob/master/static/img/_posts/Deep_Learning/no3/sigmoid_graph.png?raw=true)

## ReLU Function

ReLU 함수는 입력이 0을 넘으면 입력 그대로 출력하고, 0이하이면 0을 출력하는 함수이다.

```python
import numpy as np
import matplotlib.pylab as plt


def ReLU(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
y = ReLU(x)

print(ReLU(x))

plt.plot(x, y)
plt.show()

```

numpy의 np.maximum(0, x)는 0과 x중 큰값을 반환하는 함수이다.
![ReLU](https://github.com/Marshmellowon/marshmellowon.github.io/blob/master/static/img/_posts/Deep_Learning/no3/ReLU.png?raw=true)
