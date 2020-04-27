---
layout: post
title: "About Java Version"
date: 2020-03-29 16:37:25 +0900
categories: [Java]
sitemap:
  lastmod: 2020-04-25 17:00:00
  changefreq: daily
  priority: 1.0
---

# Java Version

###### Latest Update : {{site.time}}

### 1. 역사

- Java는 Sun Microsystems의 James Arthur Gosling과 연구원들이 개발한 객체 지향 프로그래밍 언어다. 1995년에 처음 발표하였으며, 초기에는 가전제품 내에 탑재하여 동작하는 프로그램을 위해 개발했다. 현재는 웹 어플리케이션 개발에 가장 많이 사용되는 언어중 하나이다. 현재 2020년 3월 18일에 공개된 Java SE 14가 가장 최신 버전이다.

### 2. Java Version

Version|Release date|
JDK Beta|1995
JDK 1.0| January 1996
JDK 1.1 |February 1997
J2SE 1.2| December 1998
J2SE 1.3 |December 1998
J2SE 1.4 |February 2002
J2SE 5.0 |September 2004
Java SE 6 |December 2006
Java SE 7 |July 2011
Java SE 8 (LTS) |March 2014
Java SE 9 |September 2017
Java SE 10 |March 2018
Java SE 11 (LTS) |September 2018
Java SE 12 |March 2019
Java SE 13 |September 2019
Java SE 14 |March 2020
Java SE 15 |September 2020
Java SE 16 |March 2021
Java SE 17 (LTS) |September 2021

### 3. Java Version의 성장

#### 1. JDK 1.0

- Java의 첫 버전이며, 안정적인 버전은 JDK 1.0.2이다.

#### 2. JDK 1.1

- Inner class와 JavaBeans, JDBC, RMI가 추가되었다.
- Taligent로 부터 유니코드를 지원 받았다.

#### 3. J2SE 1.2

- GUI, JIT, CORBA가 추가 되었다.

#### 4. J2SE 1.3

- HotSpot, JVM, JNDI, JPDA, JavaSound이 추가되었다.

#### 5. J2SE 1.4

- assert keyword, Non-blocking, Java Web Start가 추가되었다
- IPv6를 지원하게 되었다.
- JPEG, PNG 형식을 지원하는 Image I/O API를 추가 하였다.

#### 6. J2SE 5.0

- 버전 표기에서 앞의 1.을 빼고 표기하기 시작하였다.
- Generics, Annotation, Auto Boxing/Unboxing, Enumeration, 가변 길이 파라미터, Static Import 등이 추가되었다.
- 일반적인 지원은 2009년에 종료되었으며, 연장 지원은 2015년도에 종료되었다.

#### 7. Java SE 6

- 버전 표기가 J2SE에서 Java SE로 변경 되었다.
- Scripting Language Support, JDBC 4.0, Java Compiler API, Pluggable Annotation등이 추가되었다.
- Java 6 이후 SUN과 이후의 Oracle은 공개 API를 변경하지 않으면서 최종 사용자의 편의성을 향상 시키거나 버그를 수정한 몇 가지 업데이트를 발표했다.
- 일반지원은 2013년에 종료되었으며, 연장지원은 2018년에 종료되었다.

#### 8. Java SE 7

- JVM은 Dynamic languages를 지원 하였고, try문에서 자동 자원관리를 지원 하게 되었다.
- Switch문에서 String을 사용하게 되었다.
- Elliptic Curve Cryptography, Java2D를 위한 XRender, Upstream, Java Deployment Ruleset 등이 추가되었다.
- 일반 지원은 2015년에 종료되었으며, 연장 지원은 2022년에 종료될 예정이다.

#### 9. Java SE 8 (LTS)

- Annotation on Java Types, Unsigned Integer 계산, Repeating Annotation, 새로운 날짜와 시간 API가 추가 되었다.
- Oracle은 지속적으로 공식 업데이트와, Java SE 8의 자동 업데이트를 2020년 12월 까지 개인 사용자에게 제공하고 상업적 사용자는 2019년 1월까지 제공 한다.
- 32비트를 지원하는 마지막 공식 Java 버전이다.

#### 10. Java SE 9

- 가장 큰 특징은 Project Jigsaw 기반으로 런타임이 모듈화된 것이다.
- Java Applet기능 지원이 종료된다.
- 이 버전 부터 64비트만 지원하게 출시 되었다.
- 일반지원은 2018년에 종료 되었다.

#### 11. Java SE 10

- var 키워드를 사용한 지역 변수 타입 추론, 병렬처리 가비지 컬렉션, 개별 쓰레드로 분리된 Stop-The-World, 루트 CA 목록 등이 추가되었다.
- JDK의 repository가 하나로 통합되었고, JVM 힙 영역을 시스템 메모리가 아닌 다른 종류의 메모리에도 할당할 수 있게 되었다.
- 일반 지원은 2018년에 종료되었다.

#### 12. Java SE 11 (LTS)

- JavaFX가 JDK에서 분리되어 별도의 모듈로 제공된다.
- 람다 파라미터에 대한 지역 변수 문법 엡실론 가비지 컬렉터, HTTP 클라이언트 표준화 등의 기능이 추가되었다.
- 일반 지원은 2023년, 연장 지원은 2026년에 종료될 예정이다.

#### 13. Java SE 12

- 가비지 컬렉터 개선, 마이크로 벤치마크 툴 추가, 성능 개선이 이루어 졌다.
- Switch문을 확장한 것이 특징이다.
  기본 Switch문에서

```java
/* java 문법 명시 */

switch (day) {
    case MONDAY, FRIDAY, SUNDAY -> System.out.println(6);
    case TUESDAY                -> System.out.println(7);
    case THURSDAY, SATURDAY     -> System.out.println(8);
    case WEDNESDAY              -> System.out.println(9);
}
```

위의 형식으로 사용할 수 있게 되었다.

#### 14. Java SE 13

- 추가적인 Switch문의 개선이 이루어 졌다.

```java
/* java 문법 명시 */
var a = switch (day) {
    case MONDAY, FRIDAY, SUNDAY -> yield 6;
    case TUESDAY                -> yield 7;
    case THURSDAY, SATURDAY     -> yield 8;
    case WEDNESDAY              -> yield 9;
};
```

yield라는 예약어가 추가되었다.

#### 15. Java SE 14

- 2020년 3월 18일에 출시 되었다.
