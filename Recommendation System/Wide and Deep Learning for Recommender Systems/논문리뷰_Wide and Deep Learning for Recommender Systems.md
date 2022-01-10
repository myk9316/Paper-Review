# Wide & Deep Learning for Recommeder Systems
본 논문은 2006년 구글에서 발표한 wide and deep 추천랭킹 알고리즘에 관한 논문이다. 이 알고리즘은 구글플레이에서 앱 추천에 실제로 적용된 알고리즘으로, 선형 모델과 신경말 모델을 함께 적용하여 Memorization과 Generalization 둘다를 얻어 추천랭킹을 개선하는 것이 목표이다. 

<br/>

## ABSTRCT
Generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification problems with sparse inputs. Memorization of feature interactions through a wide set of cross-product feature transformations are effective and interpretable, while generalization requires more feature engineering effort. With less feature engineering, deep neural networks can generalize better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features. However, deep neural networks with embeddings can over-generalize and recommend less relevant items when the user-item interactions are sparse and high-rank. In this paper, we present Wide & Deep learning—jointly trained wide linear models and deep neural networks—to combine the benefits of memorization and generalization for recommender systems. We productionized and evaluated the system on Google Play, a commercial mobile app store with over one billion active users and over one million apps. Online experiment results show that Wide & Deep significantly increased app acquisitions compared with wide-only and deep-only models. We have also open-sourced our implementation in TensorFlow.

- Sparse input인 경우, 회귀와 분류 문제를 풀기 위해 주로 linear model를 사용한다. 
- 이 때, wide한 cross-product feature transformation은 feature interaction을 암기하는데는 효과적이고 해석 가능하지만, 일반화에 더 많은 피처 엔지니어링이 필요하다. 
- 반면, sparse feature에 대해 학습한 저차원 임베딩을 활용한 deep neural network는 피처 엔지니어링에 적은 effort가 들고, 전에 나오지 않았던 변수의 조합에 대한 일반화를 더 잘할수있다. 하지만, deep neurla network는 사용자-아이템 상호작용이 sparse하거나 high-rank이면, 과대적합 시키고 관련이 적은 아이템을 추천해줄 수 있다. 
- 이 논문은 암기와 일반화의 장점을 조합하기 위해 wide한 선형모형과 deep한 신경망을 함께 훈련시키는 방법을 제안하며, 해당 방법은 실제로 Google Play에 적용이 되어서 앱 가입을 크게 증가시켰다. 

<br/>

## 1. Introduction
- 추천시스템은 사용자 정보 및 맥락 정보가 input query 이고, 아이템의 순위 리스트가 output query라는 걸로 볼 때 검색 순위 시스템으로 생각할 수 있다. 추천 작업은 query가 주어졌을 때, 데이터베이스에서 관련 있는 아이템을 찾고, 클릭 또는 구매 같은 특정 목표에 따라 아이템의 순위를 매긴다. 

- 추천시스템이 마주하는 어려운 점 중 한가지는 암기(Memorization)와 일반화(Generalization)를 동시에 달성하는 것이다. 

- 본 논문에서는 선형 모델과 신경망을 결합해서 학습함에 따라, 암기와 일반화를 한 모델에서 달성하여 각각의 이점을 결합하였다. 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/148770417-18c60700-98fa-4be0-94a0-e6b18932b880.PNG"></p>


### 암기(Memorization) - Wide한 영역 담당
- 암기는 동시에 발생하는 아이템 또는 피처를 학습하고 사용 가능한 과거 데이터로부터 상관관계를 추출하는 작업으로, 암기에 기반한 추천은 사용자가 실행한 행동과 관련된 아이템과 직접적으로 관련이 있다.

- Linear model을 사용하며, 동시출현 빈도를 표현하는 cross-product를 통해 적은 parameter로도 모든 feature의 조합을 기억할 수 있어 wide하고 sparse한 정보 기억이 효과적이다. 

- sparse feature (ex. AND(user_installed_app=netflix, imporession_app=pandora)에 대해 **cross-product transformation** 사용하여 효과적으로 달성할 수 있다. 덜 세분화된 변수(ex. AND(user_installed_category=video, imporession_category=music)를 사용해서 일반화 할 수 있지만, **피처 엔지니어링에 많은 노력**이 필요하다. 

- 단점: cross-product transformation의 한계는 학습 데이터에 나타나지 않은 query-item 변수 쌍은 학습하지 못하고, 뻔한 추천을 한다. 또한, Wide의 일반화는 feature engineering이 많이 필요하다. 

### 일반화(Generalization) - Deep한 영역 담당
- 일반화는 상관관계의 이행성에 기반하고, 과거에 전혀 혹은 드물게 발생한 새로운 변수들의 조합을 탐구하여 추천의 다양성을 향상시키려는 경향이 있다. (비주류 아이템을 거의 추천하지 않는 Long-tail problem 극복에 도움) 

- Deep neural netowrk를 사용하는데, 피처 엔지니어링에 적은 effort가 들어가며 non-linear한 output을 내기 때문에 이전에 나타나지 않은 변수들에 대해서도 연관성을 학습 시킬 수 있다. 

- Embedding based 모델(FM or deep neural network)은 저차원 임베딩 벡터의 학습을 통해 이전에 보지 못했던 query-item 쌍을 **피처 엔지니어링을 줄이면서** 일반화를 할 수 있다.
  - 단점: 특정 선호도를 가진 사용자나 틈새 아이템과 같이 sparse하고 high-rank인 경우에는 query-item 행렬에 대해 저차원 표현으로 학습하는 것은 어렵다. 즉, 실제로 존재할 수 없거나 희소한 관계에 대해서도 지나친 일반화를 하여 관련이 적은 추천이 이루어질 수 있다. 

### Contribution
- Sparse input에 대한 추천시스템을 위해 임베딩을 통한 피드포워드 신경망과 변수 변환을 통한 선형 모델을 함께 훈련 시키는 Winde & Deep 모델
- 10억명 이상의 활성 사용자와 100만개 이상의 앱이 있는 모바일 앱스토어 구글 플레이 에서 Wide & Deep 추천시스템의 구현 및 평가
- Tensorflow 고수준 API를 통한 오픈소스화

<br/>

## 2. Recommender System Overview
- 아래 그림은 구글 스토어 추천시스템의 개요이다. 
  - 사용자가 앱 스토어에 방문하면 사용자 본인과 맥락에 관련한 다양한 피처가 포함된 query가 생성된다. 
  - 추천시스템은 사용자가 클릭이나 구매 같은 특정한 행동을 수행할 수 있는 앱 목록을 반환한다. 
  - 이러한 사용자 행동은 query와 impression과 함께 학습을 위한 훈련 데이터로 로그에 기록된다. 

- 하지만, 데이터베이스에 100만개가 넘는 앱이 있기 때문에, 요구되는 서비스 대시 시간 이내로 모든 쿼리에 대하여 모든 앱에 점수를 주는 것은 어렵다. 
  - 따라서, 사용자의 query가 들어오면 검색(retrieval)시스템은 다양한 신호(일반적으로 머신러닝 모형과 사람이 정의한 규칙으로)를 사용하여 해당 query에 가장 적합한 짧은 앱 목록을 반환한다. 
  - 이어서, Ranking 시스템은 앱 목록에 있는 모든 아이템에 대해 점수를 매긴다. 여기서 점수는 사용자의 정보인 다양한 피처들을 기반으로 사용자 x가 앱 y에 action할 확률인 P(y|x)를 구한다. 

- 본 논문은, Wide & Deep learning를 사용한 Ranking 모델을 제안한다. (앱 목록의 점수를 매기는데 사용) 


<p align="center"><img src="https://user-images.githubusercontent.com/79245484/148804633-8667463e-f7a8-42d0-a2aa-0eca72ad81f5.PNG"></p>

<br/>

## 3. Wide & Deep Learning
### 3.1 The Wide Component
- 

<br/>

### 3.2 The Deep Component
- 

<br/>

### 3.3 Joint Training of Wide & Deep Model
- 

<br/>

## 4. System Implementation
### 4.1 Data Generation
- 

<br/>

### 4.2 Model Training
- 

<br/>

### 4.3 Model Serving
- 

<br/>

## 5. Experiment Results
### 5.1 App Acquisitions
- 

<br/>

### 5.2 Serving Performance
- 

<br/>

## 6. Related Work
- 

<br/>

## 7. Conclusion
- 


