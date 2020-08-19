---
layout: post
title: "Deep Learning for Scratch(2)"
subtitle: Deep Learning Study
tags: [AI, Python]
author: marshmello
comments: True
date: 2020-07-08
lastmod: 2020-04-25 17:00:00
sitemap:
changefreq: daily
priority: 1.0
---

<sub>책 정보: [밑바닥부터 시작하는 딥러닝](https://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)</sub>

## Perceptron(퍼셉트론)

퍼셉트론 알고리즘은 1957년 프랑크 로젠블라트가 고안한 것으로 딥러닝의 기원이 되는 알고리즘이다.

퍼셉트론의 기본 원리는 여러개의 신호를 입력으로 받고 하나의 신호를 출력하는 것이다.
2개의 입력 신호를 받은 퍼셉트론은 다음과 같다.

![perceptron](/assets/img/posts/Deep_Learning/perceptron.png)

x<sub>1</sub>, x<sub>2</sub> - 입력신호
w<sub>1</sub>, w<sub>2</sub> - 가중치
y - 출력신호
b - 임계값

퍼셉트론 동작원리를 식으로 표현한 것이다.  
y = {0 (x<sub>1</sub>w<sub>1</sub> + x<sub>2</sub>w<sub>2</sub> <= b)}  
y = {1 (x<sub>1</sub>w<sub>1</sub> + x<sub>2</sub>w<sub>2</sub> > b)}

## 단순한 논리 회로

### AND 게이트

퍼셉트론으로 AND게이트를 구현할 수 있다.
AND게이트의 진리표이다.

![ANDgate](/assets/img/posts/Deep_Learning/ANDgate.JPG)

AND게이트를 코드로 구현해 보았다.

```python
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


print("AND logic")  # AND logic
print(AND(0, 0))    # 0
print(AND(1, 0))    # 0
print(AND(0, 1))    # 0
print(AND(1, 1))    # 1
```

### 가중치와 편향 도입

편향이 도입된 퍼셉트론의 식이다.  
y = {0 (b + x<sub>1</sub>w<sub>1</sub> + x<sub>2</sub>w<sub>2</sub> <= 0)}  
y = {1 (b + x<sub>1</sub>w<sub>1</sub> + x<sub>2</sub>w<sub>2</sub> > 0)}

편향을 사용하면 두 입력이 모두 0이어도 최종결과는 편향값을 출력한다.

```python
import numpy as np


def ANDbias(x1, x2):
    x = np.array([x1, x2])  # 입력
    w = np.array([0.5, 0.5])  # 가중치
    b = -0.7  # 편향

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


print(ANDbias(0, 0))    # 0
print(ANDbias(1, 0))    # 0
print(ANDbias(0, 1))    # 0
print(ANDbias(1, 1))    # 1
```

편향을 도입하여도 같은 값이 나오는 것을 확인할 수 있다.

## 정리

신경망의 기초가되는 퍼셉트론 알고리즘을 알아보았다.

##### 다음 post: [Deep Learning for Scratch(3)](https://marshmellowon.github.io/ai/python/2020/07/08/Deep_Learning_from_Scratch3.html)
