---
layout: post
title: "RNN의 변형 LSTM, GRU"
subtitle: TAVE 6TH Reinforcement Learning Study
tags: [TAVE, Deep Learning]
author: marshmello
comments: True
date: 2020-09-05
lastmod: 2021-01-15 21:52:00
sitemap:
  changefreq: daily
  priority: 1.0
use_math: false
---

# Outline

1. LSTM
2. GRU
3. seq2seq Attention

# LSTM

- LSTM은 언어, 음성인식과 같은 다양한 분야에서 사용된다.
- RNN에서 gradient vanishing을 해결할 수 있다.

## LSTM Cell

- 단순히 activation만 하는것이 아니라 시간적인 흐름을 조절해서 메모리를 잊어버릴지 유지할지 제어하는 네트워크가 심어져 있다.

- LSTM Cell은 RNN Cell의 장기 의존성 문제를 해결하며, 빠르게 학습이 가능하다.
- RNN과 같은 체인구조를 가지고 있지만, 반복모듈의 구조가 다르다.

> 3gates : f<sub>t</sub>, i<sub>t</sub>, o<sub>t</sub>  
> 2 outputs: h<sub>t</sub>, C<sub>t</sub>  
> 4 params: W<sub>i</sub>, W<sub>f</sub>, W<sub>o</sub>, W<sub>h</sub>

![LSTMcell](/assets/img/LSTM_GRU/LSTM_Cellmy.png)  
 <sub>이미지 참고: [colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)</sub>

## Forget Mechanism

- 과거의 정보를 기억할지 잊을지를 결정하는 단계이다.
- forget mechanism은 hidden unit(h<sub>t-1</sub>)과 input을 받아 sigmoid를 통과한다
- f<sub>t</sub> = 𝜎(W<sub>f</sub> x \[h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>f</sub>)
  ![forget](/assets/img/LSTM_GRU/forget.png)

## Input Mechanism

- 현재의 정보를 잊을지 기억할지 결정하는 gate이다.
- input mechanism은 hidden unit(h<sub>t-1</sub>)과 input을 받아 sigmoid와 tanh를 통과 하여 C<sub>t</sub>를 생성한다.
- i<sub>t</sub> = 𝜎(W<sub>i</sub> x \[h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>i</sub>)
  ![input](/assets/img/LSTM_GRU/input.png)

## Output Mechanism

- 최종적인 출력값이다. 현 시점의 hidden state는 현 시점의 cell state와 계산되어 출력되며, 다음 시점으로 hidden state를 넘긴다.
- o<sub>t</sub> = 𝜎(W<sub>o</sub> x \[h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>o</sub>)
- h<sub>t</sub> = o<sub>t</sub> \* tanh(C<sub>t</sub>)
- C<sub>t</sub> = f<sub>t</sub> \* C<sub>t-1</sub> + i<sub>t</sub> \* C'<sub>t</sub>
- C'<sub>t</sub> = tanh(W<sub>C</sub> x \[h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>C</sub>)
  ![output](/assets/img/LSTM_GRU/output.png)

# GRU Cell

- LSTM cell의 간소화된 버전이다.
- LSTM에서 두 output C<sub>t</sub>와 h<sub>t</sub>가 하나의 벡터 h<sub>t</sub>로 통합되었다.
- z<sub>t</sub>가 forget과 input 게이트를 모두 제어한다.
  - z<sub>t</sub>가 1일때 forget 게이트가 열리고, 0일때 input 게이트가 열린다.
- output게이트가 없으며, 이전 hidden cell에서 어느 부분이 출력될지 결정하는 것은 r<sub>t</sub>가 수행한다.
  ![gru](/assets/img/LSTM_GRU/gru.png)

### Reset Gate

- 두 번째 식에 해당하는 부분이다. 이전 시점의 hidden state와 현 시점의 x를 𝑠𝑖𝑔𝑚𝑜𝑖𝑑를 적용하여 구한다. 즉 이전 hidden state를 얼마나 활용할지 결정한다.

### Update Gate

- 과거와 현재의 정보를 얼마나 반영할지 비율을 정하는 부분이다. 𝑧는 현재의 정보에 대한것, 1−𝑧는 과거 정보에 대한 반영 비율을 나타낸다.

# seq2seq Attention

- seq2seq은 입력 시퀸스를 받아 출력 시퀸스를 생성하는 모델이다. 가변적인 입 출력 시퀸스 길이를 처리할 수 있다. 하지만 vanishing gradient와 정보의 손실 문제가 발생한다. (bottleneck problem)
- 특정한 정보를 전달하기 위해 Attention Mechanism 사용한다.
- 디코더의 매 시점마다 인코더의 전체 입력을 다시 한번 참조할 수 있게 도와주는 역할을 한다.
- 전체 입력을 동일하게 참조하는 것이 아닌 현재 예측해야 할 부분과 가장 연관있는 부분에 더 집중적으로 참조한다.

### <b>\[Score Function]</b>

- Decoder의 hidden space vector d1을 구한다.
- d1은 Encoder의 모든 hidden space vector를 자신의 위치와 가장 관계가 깊은 단어가 무엇인지 고민한다.
- 모든 e를 살펴보며 score funtion을 통과 한 것이 Attention Score이다.

![seq2](/assets/img/LSTM_GRU/seq2.png)

### <b>\[Normalize]</b>

- Sorce에 Softmax를 취한다.
- e와 a를 가지고 context vector c를 만든다.

![seq3](/assets/img/LSTM_GRU/seq3.png)

### <b>\[다음 언어 예측]</b>

- c와 d를 가지고 다음 언어를 예측할 수 있다. 이 때, 단어 예측을 위한 f는 Neural Network를 사용한다.

  ![seq4](/assets/img/LSTM_GRU/seq4.png)
