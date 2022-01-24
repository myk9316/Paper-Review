# BPR: Bayesian Personalized Ranking from Implicit Feedback
본 논문은 2009년에 발표된 논문으로, 

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
  - 관측되지 않은 데이터는 negative feedback(실제로 사용자가 구매에 관심이 없음) 또는 missing value(사용자가 미래에 구매할 수 있음)이며, 구분이 안된다. 
  - Non-observed user-item pair = real negative feedback + missing value
  - Implicit 데이터를 모델링할 때는 수집되지 않은 데이터도 같이 모델링해야 한다. 

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
- 앞서 설명했듯이 implicit feedback은 긍정적인 클래스만 관측되며, 관측되지 않은 데이터(real negative feedback + missing value)도 모두 고려하여 모델링을 한다. 

- 하지만, 기존 item recommendation의 머신러닝 접근 방법은 관측된 데이터는 1로, 관측되지 않은 데이터는 0으로 표시된다. (1은 선호함, 0은 선호하지 않음)
  - 이러한 방법의 문제점은 missing value를 모두 negative feedback으로 간주함에 따라, 미래에 선호할 수도 있는 아이템들이 모두 무시되고 0으로 표기된다. 
  
  - 따라서, 사용자가 실제로 구매할지 모르는 아이템들도 0으로 예측되는 문제가 발생한다. (머신러닝 모델이 구분을 잘 못하게됨) 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/150670074-ee9b1225-9a5d-464f-9004-1b60148b6b53.PNG" width="40%" height="40%"></p>

- 따라서, 저자는 아래와 같은 방법을 사용하여 기존의 방법과 다른 방식으로 문제를 해결한다. <p align="center"><img src="https://user-images.githubusercontent.com/79245484/150670076-f6ca18cb-4885-4442-b890-4a2835aeb57d.PNG" width="40%" height="40%"></p>
  - 단순히 missing value를 negative feedback으로 간주하는 방법보다 문제를 더 잘 표현하는 방법으로, 단일 아이템에 점수를 매기는 대신에 두 아이템 pair의 랭크를 매기는 것으로 데이터셋을 가공한다. (관측되지 않은 item에도 정보를 부여해 간접적으로 학습시킬 수 있음) 
  - 가정
    - 사용자는 관측된 아이템을 관측되지 않은 아이템들보다 선호한다. (item i가 관측되었고 item j가 관측되지 않았다면, item i를 더 선호한다)
    
    - 관측된 아이템들 간에는 선호도를 추론할 수 없다. (item i와 j가 모두 관측되었으면, 어떤 아이템을 더 선호하는지 알 수 없다)
    
    - 관측되지 않은 아이템들 간에는 선호도를 추론할 수 없다. (item i와 j가 모두 관측되지 않았으면, 어떤 아이템을 더 비선호하는지 알 수 없다) 
    
  - +는 사용자가 item i를 item j에 비해 선호하는 것을 뜻하고, -는 사용자가 item j를 item i에 비해 선호하는 것을 뜻한다. 
  - 학습 데이터를 다음과 같이 표시할 수 있다. <p align="center"><img src="https://user-images.githubusercontent.com/79245484/150670075-e028bb54-7bba-49ab-aef6-434ec8ef8a24.PNG" width="40%" height="40%"></p>

- 이러한 방식은 장점은 다음과 같다.
  - 학습 데이터는 positive 뿐만 아닌, positive, negative, missing value로 구성되며, 아이템쌍(i,j)에서 관측되지 않은 missing value는 추후에 랭크가 매겨져야하는 테스트 데이터셋이 된다. 즉, 학습 데이터와 테스트 데이터가 서로 disjoint(상호 배타적)임
  
  - 학습 데이터는 실제 랭킹 목적에 맞게 만들어지고, 관측된 데이터의 부분집합인 D<sub>s</sub> 는 학습 데이터로 사용된다. 

<br/>

## 4. Bayesian Personalized Ranking (BPR)
- 앞에서 정의한 학습 데이터 D<sub>s</sub>로 Bayesian Personalized Ranking을 구하는 방법을 소개한다.
- <img src="https://latex.codecogs.com/svg.image?p(i>&space;_{u}&space;j|\Theta&space;)" title="p(i> _{u} j|\Theta )" />에 대한 likelihood function과 model parameter <img src="https://latex.codecogs.com/svg.image?p(\Theta)" title="p(\Theta)" />에 대한 prior probability를 사용하는 베이지안 문제

<br/>

### 4.1 BPR Optimization Criterion
- 베이지안 optimization의 목적은 아래의 사후확률을 최대화 하는 파라미터 <img src="https://latex.codecogs.com/svg.image?\Theta" title="\Theta" />를 찾는 것이다. (최대 사후 확률  추정, MAP) <p align="center"><img src="https://user-images.githubusercontent.com/79245484/150670077-ce9320dd-846f-4c59-b9bb-74e8470aa641.PNG" width="30%" height="30%"></p>

#### Likelihood
- 모든 사용자는 독립적이고, 특정 사용자의 아이템에 대한 랭킹은 독립적이라고 가정한다. 따라서, user-specific likelihood function은 다음과 같이 D<sub>s</sub>에 포함되는(특정 사용자 u의 item i의 랭크가 item j의 랭크보다 높은 경우)와 포함되지 않은 경우의 곱으로 표현할 수 있다. <p align="center"><img src="https://user-images.githubusercontent.com/79245484/150670078-5c1b5903-c4bd-4d9a-85dd-158ab09c1471.PNG" width="50%" height="50%"></p> <p align="center"><img src="https://user-images.githubusercontent.com/79245484/150670079-be108125-82df-4ad9-869c-6167a05d4cc8.PNG" width="30%" height="30%"></p>  

- 위 수식은, Totality와 antisymmetry에 따라서 다음과 같이 D<sub>s</sub>의 모든 경우를 곱하는 것으로 simplified할 수 있다. <p align="center"><img src="https://user-images.githubusercontent.com/79245484/150670081-fc3e590d-b8a9-4e78-9814-39da7d17f7d1.PNG" width="40%" height="40%"></p> 

- 아직까지는 사용자 각각의 아이템에 대한 랭킹이 보장되지 않으므로, 다음과 같이 각 사용자의 아이템 (i,j)에 대한 선호 확률을 정의한다. 여기서 x<sub>uij</sub>는 MF나 knn으로 구한다. <p align="center"><img src="https://user-images.githubusercontent.com/79245484/150670082-8db1cbf5-7e0a-4848-ae2c-6222ff95e784.PNG" width="30%" height="30%"></p> <p align="center"><img src="https://user-images.githubusercontent.com/79245484/150670083-250d7aea-d88e-4d86-9afe-f383f26e630e.PNG" width="20%" height="20%"></p> 


#### Prior probability
- 사전 확률 분포를 구하기 위해 general prior density를 다음과 같이 정의한다. (mean 0, variance-covariance matrix) <p align="center"><img src="https://user-images.githubusercontent.com/79245484/150670085-4f0c6a22-3a08-49a1-acd9-82d384305648.PNG" width="20%" height="20%"></p>, <p align="center"><img src="https://user-images.githubusercontent.com/79245484/150720514-c2b15ccb-19a6-41e2-95b9-5e441a1afd6a.PNG" width="15%" height="15%"></p>


#### BPR-OPT
- 최종 BPR-OPT의 maximum posterior estimator 형태는 다음과 같다. <p align="center"><img src="https://user-images.githubusercontent.com/79245484/150670087-66e22a30-6d5c-467c-b743-937f00031c7a.PNG" width="50%" height="50%"></p> 

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
