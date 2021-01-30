---
layout: post
title: "Neural Network VI"
subtitle: convolutional network
tags: [TAVE, TAVE Research, Deep Learning]
author: marshmello
comments: True
date: 2021-01-30
lastmod: 2021-01-30 11:52:00
sitemap:
  changefreq: daily
  priority: 1.0
use_math: true
---

# Convolutional Neural Network

- 이미지 인식과 음성 인식 등 다양한 곳에 사용된다.

## Structure

![/assets/img/posts/TaveResearch/neuralN6/structure.png](/assets/img/posts/TaveResearch/neuralN6/structure.png)

- Convolutional layer(합성곱 계층) 과 Down sampling layer로 구성되어 있다.
- Convolutional layer는 입력의 패턴을 스캔하는 뉴런으로 구성되어 있다.

## Problem of fully-connected layers

- 이전까지 fully-connected Networks에서는 Affine layer(완전 연결 계층)를 사용했다.
- Affine layer의 문제점은 데이터의 형상이 무시된다는 것이다. 3차원(세로, 가로, 채널)로 구성되어 있는 3차원 이미지를 Affine layer에 입력하기 위해서는 1차원 데이터로 변환해 줘야 한다.
- Affine layer는 형상을 무시하고 모든 입력 데이터를 같은 차원의 뉴런으로 취급하여 형상이 유지되지 않는다.

### Convolution Operation

![/assets/img/posts/TaveResearch/neuralN6/conv1.png](/assets/img/posts/TaveResearch/neuralN6/conv1.png)

- Convolution layer에서의 합성곱(convolution) 연산을 수행한다.
- 이미지 처리에서 필터 연산과 같다.
- 필터를 커널(kernel)이라고도 한다.
- 필터가 왼쪽 위 부터 스캔하며 합성곱 연산을 진행하면 output과 같은 값들이 나온다.

## Padding

![/assets/img/posts/TaveResearch/neuralN6/padding.png](/assets/img/posts/TaveResearch/neuralN6/padding.png)

- 합성곱 연산을 진행하기 전, 입력 데이터 주변을 0으로 채우는 것을 Padding이라고 한다.
- (5X5) 크기의 데이터에 폭이 1인 padding을 적용한 후, (3X3)크기의 필터로 스캔을 하면 (5X5) 크기의 출력 데이터가 나온다.
- 연속적으로 합성곱 연산을 하다보면 크기가 작아지기 때문에 Padding을 사용하여 출력 데이터의 크기를 조절하는 용도로 사용된다.

## Stride

![/assets/img/posts/TaveResearch/neuralN6/stride.png](/assets/img/posts/TaveResearch/neuralN6/stride.png)

- Stride는 필터를 적용하는 위치의 간격을 말한다.
- 위 그림은 stride를 2로 적용시킨 것이다.
- Stride가 커질수록 출력의 크기는 작아진다.

## Output size

![/assets/img/posts/TaveResearch/neuralN6/output.png](/assets/img/posts/TaveResearch/neuralN6/output.png)

- output size를 계산하는 식이다.
- 입력 데이터가 (HXW)이면 input size에 H와 W가 들어간다.
- 출력의 크기가 정수가 아니면 오류를 내거나 반올림하여 진행하도록 구현하는 경우도 있다.

---

## Pooling layer

- 풀링은 가로 세로 방향의 공간을 줄인다.
- Convolutional layer와 달리 학습해야 할 매개변수가 없다.
- 입력된 데이터의 채널 그대로 출력 데이터로 내보낸다. 채널의 갯수가 변하지 않는다.
- 입력 데이터의 변화에 영향을 크게 받지 않는다.

### Max Pooling

![/assets/img/posts/TaveResearch/neuralN6/max.png](/assets/img/posts/TaveResearch/neuralN6/max.png)

- Max pooling은 위 그림과 같이 (2x2)영역에서 가장 큰 값을 선택하여 출력한다.
- 이미지 인식에서는 주로 max pooling을 사용한다.

### Average Pooling

![/assets/img/posts/TaveResearch/neuralN6/avg.png](/assets/img/posts/TaveResearch/neuralN6/avg.png)

- Average pooling(mean pooling)은 (2x2)영역에서 평균값을 출력으로 가진다.
