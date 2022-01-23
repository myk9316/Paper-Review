# BPR: Bayesian Personalized Ranking from Implicit Feedback


<br/>

## Abstract
Item recommendation is the task of predicting a personalized ranking on a set of items (e.g. websites, movies, products). In this paper, we investigate the most common scenario with implicit feedback (e.g. clicks, purchases). There are many methods for item recommendation from implicit feedback like matrix factorization (MF) or adaptive knearest- neighbor (kNN). Even though these methods are designed for the item prediction task of personalized ranking, none of them is directly optimized for ranking. In this paper we present a generic optimization criterion BPR-Opt for personalized ranking that is the maximum posterior estimator derived from a Bayesian analysis of the problem. We also provide a generic learning algorithm for optimizing models with respect to BPR-Opt. The learning method is based on **stochastic gradient descent with bootstrap sampling.** We show how to apply our method to two state-of-the-art recommender models: matrix factorization and adaptive kNN. Our experiments indicate that for the **task of personalized ranking** our **optimization method outperforms the standard learning techniques for MF and kNN**. The results show the importance of optimizing models for the right criterion.

<br/>

## 1. Introduction
- 본 논문에서는 item recommendation을 다루며, item recommendation은 item에 대한 user-specific 랭킹을 create한다.  

- 아이템의 대한 사용자의 선호도는 사용자의 과거 행동을 통해서 학습한다. 

- 현실에서 대부분의 feedback은 explicit이 아닌 implicit이다.

- Implicit feeback은 explicit과 달리 사용자가 명시적으로 취향을 표현하지 않아도 수집할 수 있기 때문에 구하기가 쉽다. (ex. 클릭 모니터링, 조회 시간, 구매)

- 저자가 정의한 ranking을 추천하기 위한 optimization은 다음과 같다
  - item i와 j가 있고, user가 item i보다 j를 더 선호한다면 item i > item j
  - 이때, 학습할 파라미터를 최적화 하는 것이 목표 

<br/>


## 2. Contribution
- Maximum Posterior Estimator(베이지안 추론, MAP)에 기반한 최적화 기법인 BPR-Opt를 제안한다. 

- BPR-Opt를 최대화 하기 위해서 LearnBPR을 제안하며, 이 알고리즘은 boostrap sampling을 통한 Stochastic gradient descent를 사용한다. 

- LearnBPR을 MF, KNN에 적용하는 방법을 보여준다.

- BPR-opt가 다른 기법들에 비해 personalized ranking에 더 좋은 성능을 보인다. 

<br/>

## 3. Personalized Ranking
- 본 논문은 implicit feedback을 사용하여 personalized ranking을 구하는 것을 다룬다. 

- Personalized ranking은 item recommendation이라도도 부르며, 사용자에게 ranked list of item을 제공한다. 

- Implicit feedback의 특징
  - 부정적인 데이터가 관측되지 않고, 긍정적인 데이터만 관측된다. 
  - 관측되지 않은 데이터는 negative feedback(실제로 사용자가 구매에 관심이 없음) 또는 missing value(사용자가 미래에 구매할 수 있음)이며, 구분되지 않는다. 
  - Non-observed user-item pair = real negative feedback + missing value

<br/>

### 3.1 Formalization
- U는 모든 사용자 집합, I는 모든 아이템 집합이다. 
- 추천시스템의 task는 각 유저에 대해 personalized total ranking ( <img src="https://latex.codecogs.com/svg.image?>_{u}\subset&space;I^{2}" title=">_{u}\subset I^{2}" /> )을 제공하는 것이다. 
- <img src="https://latex.codecogs.com/svg.image?>_{u}" title=">_{u}" />는 다음과 같은 속성들을 만족해야 한다. 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/150670072-c42c48e9-c2c7-4449-8915-f5e9e8250925.PNG" width="50%" height="50%"></p>

- 또는 이렇게 표시할 수 있다. 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/150670073-a6b6c6c0-4d8a-4125-930e-3808218e2ab5.PNG" width="30%" height="30%"></p>


<br/>

### 3.2 Analysis of the problem setting
- 

<br/>


## 4. Bayesian Personalized Ranking (BPR)
- 

<br/>

### 4.1 BPR Optimization Criterion
- 

<br/>

#### 4.1.1 Analogies to AUC optimization
- 

<br/>

### 4.2 BPR Learning Algorithm
- 

<br/>

### 4.3 Learning models with BPR
- 

<br/>

#### 4.3.1 Matrix Factorization
- 

<br/>

#### 4.3.2 Adaptive k-Nearest-Neighbor
- 

<br/>

## 5. Relations to other methods
### 5.1 Weighteed Regularized Matrix Factorization (WR-MF)
- 

<br/>

### 5.2 Maximum Margin Matrix Factorization (MMMF)
-

<br/> 

## 6. Evaluation
-

<br/>

### 6.1 Datsets
-

<br/>

### 6.2 Evaluation Methodology
-

<br/>

### 6.3 Results and Discussion
-

<br/>

### 6.4 Non-personalized ranking
-

<br/>

## 7. Conclusion
- 
