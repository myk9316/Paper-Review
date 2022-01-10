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

- 본 논문에서는 선형 모델의 요소와 신경망의 요소를 결합해서 학습함에 따라, 암기와 일반화를 한 모델에서 달성하였다. 

## 암기(Memorization)
  - 동시에 발생하는 아이템 또는 피처를 학습하고 사용 가능한 과거 데이터로부터 상관관계를 추출하는 작업이다.
  - 암기에 기반한 추천은 사용자가 실행한 행동과 관련된 아이템과 직접적으로 관련이 있다.
  - 장단점: sparse feature (ex. AND(user_installed_app=netflix, imporession_app=pandora)에 대해 cross-product transformation 사용하여 효과적으로 달성할 수 있지만, cross-product transformation의 한계는 학습 데이터에 나타나지 않은 query-item 변수 쌍은 일반화할 수 없다. 

## 일반화(Generalization)
  - 상관관계의 이행성에 기반하고, 과거에 전혀 혹은 드물게 발생한 새로운 변수들의 조합을 탐구한다.
  - 추천의 다양성을 향상시키려는 경향이 있다. 
  - 장단점: 덜 세분화된 변수(ex. AND(user_installed_category=video, imporession_category=music)를 사용해서 일반화 할 수 있지만, 피처엔지니어링에 많은 노력이 필요하다. 
    - Embedding based 모델(FM or deep neural network)은 저차원 임베딩 벡터의 학습을 통해 이전에 보지 못했던 쿼리-아이템 쌍을 피처엔지니어링을 줄이면서 일반화를 할 수 있지만, 특정 선호도를 가진 사용자나 틈새 아이템과 같이 sparse하고 차원이 높은 경우에는 쿼리 아이템 행렬에 대해 저차원 표현으로 학습하는 것은 어렵다. 이러한 경우에, 과도한 일반화를 하여 관련없는 추천이 이루어질 수 있다. 

<br/>

## 2. Recommender System Overview
- 

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


