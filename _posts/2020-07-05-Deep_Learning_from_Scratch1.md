---
layout: post
title: "Deep Learning for Scratch(1)"
subtitle: Deep Learning Study
tags: [AI, Python]
author: marshmello
comments: True
date: 2020-07-05
lastmod: 2020-04-25 17:00:00
sitemap:
  changefreq: daily
  priority: 1.0
---

<sub>책 정보: [밑바닥부터 시작하는 딥러닝](https://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)</sub>

## Numpy

numpy는 외부 라이브러리이다. numpy를 사용하기위해 impot를 해준다.

```python
import numpy as np
```

배열을 생성하고 산술연산이 가능하다.

```python
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x * 2) # [2. 4. 6.]
print(x + y) # [3. 6. 9.]

```

## matplotlib

matplotlib 라이브러리는 그래프 그리기와 이미지 표시, 데이터 시각화 하기 쉬워진다.

```python
# 데이터 준비
x1 = np.arange(0, 6, 0.1)
y1 = np.sin(x1)
y2 = np.cos(x1)

# 그래프 그리기
plt.plot(x1, y1, label="sin")
plt.plot(x1, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
plt.legend()
plt.show()
```

위 코드를 실행한 그래프이다.  
![cos&sin](/assets/img/posts/Deep_Learning/cossin.png)

이미지를 표시하려면 matplotlib의 image에서 imread를 import하여 사용한다.

```python
from matplotlib.image import imread

img = imread('img/germany.png')

plt.imshow(img)
plt.show()
```

##### 다음 post: [Deep Learning for Scratch(2)](https://marshmellowon.github.io/2020/07/08/Deep_Learning_from_Scratch2.html)
