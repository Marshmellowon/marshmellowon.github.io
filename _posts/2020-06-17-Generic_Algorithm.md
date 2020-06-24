---
layout: post
title: "Generic Algorithm(Tesla Stock Price)"
date: 2020-06-17 14:30:25 +0900
categories: [Python, Algorithm]
sitemap:
lastmod: 2020-04-25 17:00:00
changefreq: daily
priority: 1.0
---

## 유전 알고리즘이란?

유전 알고리즘은 다윈의 '적자생존'의 개념을 최적화 문제를 해결하는데 적용한 것이다. 생물체가 적응하면서 진화해가는 것을 모방하여 최적해를 찾는 방법이다.

## 알고리즘 구성

1. 초기 후보해 집합을 생성
2. 선택 연산
3. 교차 연산
4. 돌연변이 연산

## 사용할 데이터셋

- Kaggle에서 Tesla Stock Price 데이터를 사용하였다.  
  [Tesla Stock Price(Kaggle)](https://www.kaggle.com/rpaguirre/tesla-stock-price)

- Open 열과 High열, Low열을 사용하였다. Open 가격에 따른 High와 Low의 평균값을 나타내었다.

## 유전 알고리즘으로 예측하기

### 데이터 가져오기

- Tesla Stock Price의 csv 파일의 값을 읽어 배열에 저장하는 코드이다.  
  arr에는 Open 가격을 넣고 arr2와 arr3에 각각 High와 Low값을 가져온 후 평균값을 avg 배열에 저장 하였다.

- 저장한 데이터의 형식은 float형을 사용하였다.

```python
# ----------------주어진 데이터셋--------------------
f = open('Tesla.csv', 'r')
rdr = csv.reader(f)

arr = []
arr2 = []
arr3 = []

for line in rdr:
    # Open
    arr.append(float(line[1]))
    # High
    arr2.append(float(line[2]))
    # Low
    arr3.append(float(line[3]))

# High와 Low의 평균값
avg = []
for i in range(len(arr2)):
    av = (arr2[i] + arr3[i]) / 2
    avg.append(float(av))

```

### 초기 후보해 집합을 생성

- 먼저 초기 후보해 집합을 구한다.
- Open가격 최소와 최대, 평균가격의 최소와 최대를 지정한다.

```python
Openmin = 2
Openmax = 300
Avgmin = 2
Avgmax = 300
```

- 4개의 값을 받아와 기울기 a의 값 4개를 얻는다.

```python
# 초기 식 y = ax
def expression(t, h):
    return h / t
```

```python
# 초기 a 4개 구하기
def init(Open1, Open2, Avg1, Avg2):
    a = []
    for i in range(4):
        OPEN = rn.randint(Open1, Open2)
        AVG = rn.randint(Avg1, Avg2)

        a.append(math.ceil(expression(OPEN, AVG)))
    return a
```

### 선택 연산

- 각각의 값에 대해 비율을 지정하고 랜덤값과 비교하여 새로운 배열에 넣는다.

```python
# 선택 연산
def selection(Aarr):
    ratio = []
    for i in range(4):
        if i == 0:
            ratio.append(Aarr[0] / sum(Aarr))
        else:
            ratio.append(ratio[i - 1] + Aarr[i] / sum(Aarr))

    sx = []
    for i in range(4):
        p = rn.random()
        if p < ratio[0]:
            sx.append(Aarr[0])
        elif p < ratio[1]:
            sx.append(Aarr[1])
        elif p < ratio[2]:
            sx.append(Aarr[2])
        else:
            sx.append(Aarr[3])
    return sx
```

### 교차 연산

- 교차 연산을 하기 위해 선택연산에서 나온 값들을 이진수로 변경해 준다.
- bin(x) 함수의 값은 "0b0100"이므로 0으로 대체해 준다.

```python
# bin() format
def binformat(x):
    binformat = '{0:>8}'.format(bin(x)).replace("b", "0").replace(" ", "0").replace("-", "0")
    return binformat


# int to binary
def int2Bin(str):
    binlist = []
    for i in range(4):
        binsrting = binformat(str[i])
        binlist.append(binsrting)
    return binlist

```

- 교차 연산은 각 염색채들을 교차하여 새로운 해집단을 생성한다.

```python
# 교차 연산
def crossover(arr):
    strarr = []
    for i in range(2):
        bita = str(arr[i])
        bitb = str(arr[i + 1])

        strarr.append(bita[:5] + bitb[5:])
        strarr.append(bitb[:5] + bita[5:])

    return strarr

```

### 돌연변이 연산

- invert 함수가 돌연변이 연산을 수행한다.

```python
# 돌연변이 한 값 저장하기
def mutation(mut):
    mutarr = []
    mutint = list(map(str, mut))
    for i in range(4):
        mutinv = float(invert(mutint[i]))
        mutarr.append(mutinv)
    return mutarr

```

- 돌연변이 연산은 교차된 염색체들중 랜덤한 확률로 비트가 반전이 된다.

```python
# 돌연변이 연산
def invert(char):
    ran = rn.random()
    a = int(char, 2)  # 숫자
    b = int("1000", 2)

    for i in range(5):
        p = 1 / 30
        if ran < p:
            if char[4:5] == "1":
                aa = a - b

            elif char[4:5] == "0":
                aa = a + b
            return aa
        return a

```

## 데이터 산점도

- 초기 데이터의 산점도와 예측된 값들의 산점도이다.
- pandas library를 사용하여 초기 데이터와 예측된 데이터를 DataFrame에 넣고, matlibplot library를 이용하여 산점도를 그렸다.

```python
# pandas library로 DataFrame 지정
MSEGrapg = pd.DataFrame(
    {"aValue": x, "MSE": y}
)

OpPrice = pd.DataFrame(
    {"temp": arr, "hum": avg}
)

predictgraph = pd.DataFrame(
    {"temp2": RanOpenprice, "hum2": yPredict}
)

# 그래프 그리기
plt.figure()
# 기존 데이터의 그래프를 그린다
one = plt.scatter(OpPrice['temp'], OpPrice['hum'], marker="o")
# 예측된 기울기로 그래프를 예측한다.
two = plt.scatter(predictgraph["temp2"], predictgraph["hum2"], marker="x")
plt.legend(handles=(one, two), labels=("Initial data", "predicted data"), loc="upper left")
plt.xlabel('Open_Price')
plt.ylabel('Average_Price')
plt.savefig("Tesla.png")

plt.figure()
# MSE 평가지표 그래프
MSEleg = plt.scatter(MSEGrapg['aValue'], MSEGrapg['MSE'], marker="o")
plt.xlabel('Inclination')
plt.ylabel('MSE')
plt.savefig("TeslaMSE.png")

# 그래프 띄우기
plt.show()

```

- 초기 데이터는 파란색 o모양 마커를 사용하였고, 예측된 데이터는 주황색 x모양 마커를 사용하였다.
- 최종 예측된 기울기는 1이다.

![산점도 사진.png](https://github.com/Marshmellowon/marshmellowon.github.io/blob/master/static/img/_posts/Generic_Algorithm/Tesla.png?raw=true)

## MSE(평균 제곱 오차)

- 머신러닝에서 뿐만 아니라 영상처리에서도 사용되는 추측값에 대한 정확성을 측정하는 방법이다.
- 기본식의 형태는 다음과 같다.
  <b>MSE = 1/n ∑<sup>n</sup><sub>i=1</sub>(Y'<sub>i</sub> - Y<sub>i</sub>)<sup>2</sup></b>
- MSE의 값이 작을수록 원본과의 오차가 적으며 추측한 값의 정확도가 높은 것이다.
- 아래 MSE 그래프를 보면 원래 기울기와의 오차가 클수록 MSE값이 커지는 것을 볼 수 있다.

![MSE 그래프](https://github.com/Marshmellowon/marshmellowon.github.io/blob/master/static/img/_posts/Generic_Algorithm/TeslaMSE.png?raw=true)

- 코드가 실행될 때 어떤 값들을 예측하고 지나가는지 알기위해 각 함수 출력도 같이 하였다.
- MSE의 값이 가장 작은 기울기가 1로 나왔다.

##### 출력한 값들이다.

```markdown
init: [35, 2, 4, 1]
selection: [35, 35, 35, 35]
inttobin: ['00100011', '00100011', '00100011', '00100011']
cross: ['00100011', '00100011', '00100011', '00100011']
mutationed: [35.0, 35.0, 35.0, 35.0]
MSEarr: [67913541.24469241, 135827082.48938468, 203740623.734077, 271654164.97876936]
예측된 기울기: 35.0

---

최소 MSE를 만족하는 기울기: 1.0
```

## 정리

- 유전 알고리즘을 파이썬으로 구현해 보았다. 이차함수나 지수함수가 나오는 데이터를 이용하고 싶었지만, 데이터를 찾는것이 쉽지 않았다.
- 처음에는 구현하기 힘들것이라고 생각했었다. 알고리즘 동작 구성을 이해하고, 차근차근 구현해 보니 잘 짜여졌다.
- 기상청에서 제공하는 서울시의 온도와 습도의 관계 데이터도 사용해 보았지만, Tesla Stock Price 데이터가 좀 더 정확한 것 같아 선택하게 되었다.

### code

- [Tesla Stock Price](https://github.com/Marshmellowon/Generic_Algorithm_MSE/blob/master/Tesla_StockPrice.py)
- [서울시 온도와 습도의 관계](https://github.com/Marshmellowon/Generic_Algorithm_MSE/blob/master/Generic_Alrorithm.py)

## REFERENCES

- [유전알고리즘으로 그네 타는 법 학습하기](https://www.youtube.com/watch?v=Yr_nRnqeDp0)
- [untitledtblog](https://untitledtblog.tistory.com/110)
- [morningcoding](https://morningcoding.tistory.com/entry/Python-71-ScikitLearn%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EB%B9%84%EC%84%A0%ED%98%95-%ED%9A%8C%EA%B7%80%EB%B6%84%EC%84%9D)
