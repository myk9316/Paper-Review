# Wide & Deep Learning for Recommeder Systems
본 논문은 2006년 구글에서 발표한 wide and deep 추천랭킹 알고리즘에 관한 논문이다. 구글플레이에서 앱 추천에 실제로 적용된 알고리즘으로, 선형 모델과 신경망 모델을 함께 적용하여 Memorization과 Generalization 둘다를 얻어 추천랭킹을 개선하는 것이 목표이다. 

<br/>

## ABSTRCT
Generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification problems with sparse inputs. Memorization of feature interactions through a wide set of cross-product feature transformations are effective and interpretable, while generalization requires more feature engineering effort. With less feature engineering, deep neural networks can generalize better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features. However, deep neural networks with embeddings can over-generalize and recommend less relevant items when the user-item interactions are sparse and high-rank. In this paper, we present Wide & Deep learning—jointly trained wide linear models and deep neural networks—to combine the benefits of memorization and generalization for recommender systems. We productionized and evaluated the system on Google Play, a commercial mobile app store with over one billion active users and over one million apps. Online experiment results show that Wide & Deep significantly increased app acquisitions compared with wide-only and deep-only models. We have also open-sourced our implementation in TensorFlow.

- Sparse input인 경우, 회귀와 분류 문제를 풀기 위해 주로 linear model를 사용한다. 

- 이 때, wide한 cross-product feature transformation은 feature interaction을 Memorization하는데는 효과적이고 해석 가능하지만, Generalization에 더 많은 피처 엔지니어링이 필요하다. 

- 반면, sparse feature에 대해 학습한 저차원 임베딩을 활용한 deep neural network는 피처 엔지니어링에 적은 effort가 들고, 전에 나오지 않았던 변수의 조합에 대한 Generalization을 더 잘할수있다. 하지만, deep neural network는 사용자-아이템 상호작용이 sparse하거나 high-rank이면, 과대적합 시키고 관련이 적은 아이템을 추천해줄 수 있다. 

- 이 논문은 Memorization와 Generalization의 장점을 조합하기 위해 wide한 선형모형과 deep한 신경망을 함께 훈련시키는 Wide and Deep 방법을 제안하며, 해당 방법은 실제로 Google Play에 적용이 되어서 앱 가입을 크게 증가시켰다. 

<br/>

## 1. Introduction
- 추천시스템은 사용자 정보 및 맥락 정보가 input query 이고, 아이템의 순위 리스트가 output query라는 걸로 볼 때 검색 순위 시스템으로 생각할 수 있다. 추천 작업은 query가 주어졌을 때, 데이터베이스에서 관련 있는 아이템을 찾고, 클릭 또는 구매 같은 특정 목표에 따라 아이템의 순위를 매긴다. 

- 추천시스템이 마주하는 어려운 점 중 한가지는 Memorization와 Generalization를 동시에 달성하는 것이다. 

- 본 논문에서는 선형 모델과 신경망을 결합해서 학습함에 따라, Memorization과 Generalization를 한 모델에서 달성하여 각각의 이점을 결합하였다. 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/148770417-18c60700-98fa-4be0-94a0-e6b18932b880.PNG"></p>


### Memorization - Wide한 영역 담당
- Memorization는 동시에 발생하는 아이템 또는 피처를 학습하고 사용 가능한 과거 데이터로부터 상관관계를 추출하는 작업으로, Memorization에 기반한 추천은 사용자가 실행한 행동과 관련된 아이템과 직접적으로 관련이 있다.

- Linear model을 사용하며, 동시출현 빈도를 표현하는 cross-product를 통해 적은 parameter로도 모든 feature의 조합을 기억할 수 있어 wide하고 sparse한 정보 기억이 효과적이다. 

- sparse feature (ex. AND(user_installed_app=netflix, imporession_app=pandora)에 대해 **cross-product transformation** 사용하여 효과적으로 달성할 수 있다. 덜 세분화된 변수(ex. AND(user_installed_category=video, imporession_category=music)를 사용해서 Generalization 할 수 있지만, 피처 엔지니어링에 많은 노력이 필요하다. 

- 단점
  - cross-product transformation의 한계는 학습 데이터에 나타나지 않은 query-item 변수 쌍은 학습하지 못한다. 
  - 뻔한 추천을 한다.
  - Wide의 Generalization은 feature engineering이 많이 필요하다. 

### Generalization - Deep한 영역 담당
- Generalization는 상관관계의 이행성에 기반하고, 과거에 전혀 혹은 드물게 발생한 새로운 변수들의 조합을 탐구하여 추천의 다양성을 향상시키려는 경향이 있다. (비주류 아이템을 거의 추천하지 않는 Long-tail problem 극복에 도움) 

- Deep neural netowrk를 사용하는데, 피처 엔지니어링에 적은 effort가 들어가며 non-linear한 output을 내기 때문에 이전에 나타나지 않은 변수들에 대해서도 연관성을 학습 시킬 수 있다. 

- Embedding based 모델(FM or deep neural network)은 저차원 임베딩 벡터의 학습을 통해 이전에 보지 못했던 query-item 쌍을 **피처 엔지니어링을 줄이면서** Generalization을 할 수 있다.

- 단점
  - 특정 선호도를 가진 사용자나 틈새 아이템과 같이 sparse하고 high-rank인 경우에는 query-item 행렬에 대해 저차원 표현으로 학습하는 것은 어렵다. 
  - 즉, 실제로 존재할 수 없거나 희소한 관계에 대해서도 지나친 Generalization을 하여 관련이 적은 추천이 이루어질 수 있다. 

### Contribution
- Sparse input에 대한 추천시스템을 위해 임베딩을 통한 피드포워드 신경망과 변수 변환을 통한 선형 모델을 함께 훈련 시키는 Wide & Deep 모델
- 10억명 이상의 활성 사용자와 100만개 이상의 앱이 있는 모바일 앱스토어 구글 플레이 에서 Wide & Deep 추천시스템의 구현 및 평가
- Tensorflow 고수준 API를 통한 오픈소스화

<br/>

## 2. Recommender System Overview
- 아래 그림은 구글 스토어 추천시스템의 개요이다. 
  - 사용자가 앱 스토어에 방문하면 사용자 본인과 맥락에 관련한 다양한 피처가 포함된 query가 생성된다. 
  - 추천시스템은 사용자가 클릭이나 구매 같은 특정한 행동을 수행할 수 있는 앱 목록을 반환한다. 
  - 이러한 사용자 행동은 query와 impression과 함께 학습을 위한 훈련 데이터로 로그에 기록된다. 

- 하지만, 데이터베이스에 100만개가 넘는 앱이 있기 때문에, 요구되는 서비스 대기 시간 이내로 모든 쿼리에 대하여 모든 앱에 점수를 주는 것은 어렵다. 
  - 따라서, 사용자의 query가 들어오면 검색(retrieval)시스템은 다양한 신호(일반적으로 머신러닝 모형과 사람이 정의한 규칙으로)를 사용하여 해당 query에 가장 적합한 짧은 앱 목록을 반환한다. 
  
  - 이어서, Ranking 시스템은 앱 목록에 있는 모든 아이템에 대해 점수를 매긴다. 여기서 점수는 사용자의 정보인 다양한 피처들을 기반으로 사용자 x가 앱 y에 action할 확률인 P(y|x)를 구한다. 

- 본 논문은, Wide & Deep learning를 사용한 Ranking 모델을 제안한다. (앱 목록의 점수를 매기는데 사용) 


<p align="center"><img src="https://user-images.githubusercontent.com/79245484/148804633-8667463e-f7a8-42d0-a2aa-0eca72ad81f5.PNG"></p>

<br/>

## 3. Wide & Deep Learning
### 3.1 The Wide Component
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/148809495-f2ab0298-64c3-48ac-8091-1793d9c1ffba.PNG"></p>

- Memorization을 구현하는 Wide Component는 일반화된 선형 모델의 형태인 <img src="https://latex.codecogs.com/svg.image?y&space;=&space;w^{T}x&plus;b" title="y = w^{T}x+b" />의 형태를 갖는다. (x: vector of d features, w: the model parameters, b: bias)

- 변수는 raw 입력 변수와 변환된 변수를 포함한다.

- 가장 중요한 변환 중 하나는 cross-product transformation이며, 다음과 같이 정의된다. <p align="center"><img src="https://user-images.githubusercontent.com/79245484/148812067-0a8e1788-64a9-4cb4-a6df-360a27f7ee16.PNG"></p>
  - 여기서, c<sub>ki</sub>는 i번째 변수가 k번째 변환 <img src="https://latex.codecogs.com/svg.image?\phi_{k}" title="\phi_{k}" />의 일부이면 1, 아니면 0인 boolean 변수이다. 
  
  - 이진 변수에서, cross-product transformation은 AND(gender=female, language=en)와 같이 gender=female, language=en 인 경우에는 1이고, 그렇지 않으면 모두 0이다. 
  
  - 이는, 이진 변수 사이의 상호작용을 포착하고, 일반화된 선형모델에 비선형성을 더해준다. 

<br/>

### 3.2 The Deep Component
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/148809491-bbd4cfff-8fec-4856-beae-6d268052b829.PNG"></p>

- Generalization을 구현하는 Deep Component는 피드포워드 신경망이다. 

- 범주형 변수의 경우, 원래 입력값은 문자열 변수(ex. "language=en")이다.

- 이런 sparse하고 고차원인 주형 변수 각각은 임베딩 벡터라고 불리는 저차원의 밀집한 실수 값 벡터로 변환되고, 임베딩 벡터는 임의로 초기화되고 모델 학습 중에 최종 손실 함수를 최소화 하도록 값이 훈련된다.

- 이러한 저차원의 밀집한 임베딩 벡터는 포워드 과정 중에 신경망의 hidden layers로 fed 되어진다. 구체적으로 각각의 hidden layer는 다음의 계산을 수행한다. <p align="center"><img src="https://user-images.githubusercontent.com/79245484/148812070-1d92fd1c-97cc-43a1-a6e9-946c3ba96821.PNG"></p>
  - l: layer number
  - f: activation function(ReLu)

<br/>

### 3.3 Joint Training of Wide & Deep Model
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/148809494-8662cd99-1fd5-4ee4-8c94-232babe7d71e.PNG"></p>

- Wide Component와 Deep Component는 예측치로서 각각의 output log odds의 가중치 합으로 결합되고, 그것은 joint training을 위해 하나의 공통의 로지스틱 함수로 fed 된다. 

- Joint training과 앙상블의 차이
  
  -  앙상블은 두 Model은 서로 모르는채로 개별적으로 학습되고, 그 결과값을 학습할 때는 말고 inference 단계에서만 결합되어 사용된다. 
  -  Joint training은 두 Component를 하나의 학습 과정으로 묶여있으며, 훈련시간에 합계의 가중치 뿐만 아니라 두 Component의 모든 파라미터를 동시에 optimize 시킨다. 

- Wide & Deep 모델에서의 Joint Training은 미니배치 확률적 최적화를 사용하여, output의 gradient를 wide와 deep component에 동시에 역전파하여 학습한다. 

  - 본 논문에서는, Deep part에는 Ada Grad를, Wide part에는 L1 regularization을 최적화 알고리즘으로 적용하며 Follow-the-regularized-leader(FTRL) 알고리즘을 함께 사용하였다. 

- 로지스틱 회귀 문제의 경우, 모델의 예측은 다음과 같다. <p align="center"><img src="https://user-images.githubusercontent.com/79245484/148812074-32f26fa6-db97-40bd-b616-83acc14c5ce2.PNG"></p>
  - <img src="https://latex.codecogs.com/svg.image?\sigma&space;" title="\sigma " /> : sigmoid 함수
  - <img src="https://latex.codecogs.com/svg.image?\phi(x)&space;" title="\phi(x) " /> : 초기 변수 x의 cross product transformation
  - <img src="https://latex.codecogs.com/svg.image?w_{wide}" title="w_{wide}" /> : wide 모델 전체의 가중치 벡터
  - <img src="https://latex.codecogs.com/svg.image?w_{deep}" title="w_{deep}" /> : 마지막 활성화 함수 <img src="https://latex.codecogs.com/svg.image?a^{(l_{f})}" title="a^{(l_{f})}" /> 가 적용된 가중치

<br/>

## 4. System Implementation
앱 추천 파이프라인은 데이터 생성, 모델 학습, 모델 서비스의 3단계로 구성된다. 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/148804514-aec1ba1d-bbe7-458a-a466-f3e705247fce.PNG"></p>

### 4.1 Data Generation
- 이 단계에서는 일정 시간 동안 사용자와 앱 노출 데이터를 통해 학습 데이터를 생성한다. 이때, 레이블은 '앱 가입'인데, 노출된 앱을 설치하면 1이고 그렇지 않으면 0이다.

- 범주형 문자열 변수를 정수 ID로 매핑하는 어휘 테이블도 이 단계에서 생성된다.

- 시스템은 최소 횟수 이상으로 발생하는 모든 문자열 변수에 대해 ID 공간을 계산한다. 

<br/>

### 4.2 Model Training
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/148909475-8dfad054-abae-4971-8070-17ce6f822c71.PNG"></p>

- 실험에서 사용한 모델의 구조는 위와 같다. 
  - 훈련 중, 입력 레이어는 훈련 데이터와 어휘사전을 받아서 레이블과 함께 sparse하고 dense한 변수를 생성한다. 
  
  - Wide Component는 사용자가 설치한 앱과 노출된 앱들의 cross-product transformation으로 구성
  
  - Deep Component는 32차원의 임베딩 벡터가 각각의 범주형 변수들에 대해 학습 
  
  - 모든 임베딩을 dense 변수들과 연결하여 대략 1200차원의 dense 벡터를 생성
  
  - 연결된 벡터는 3개의 RELU층으로 전달되고, 마지막으로 로지스틱 output unit으로 전달

- Wide & Deep 모델은 5천억 개가 넘는 샘플로 훈련되었는데, 새로운 학습 데이터가 도착할 때마다 모델은 다시 훈련을 해야한다. 하지만, 매번 다시 훈련시키는 것은 너무 많은 비용이 들고 데이터 수집부터 서비스까지 시간이 많이 소요된다. 이러한 문제를 해결하기 위해, 본 논문은 warm-starting system을 구현했다. 

  - warm-starting system: 이전 모델로부터 온 임베딩과 선형 모델의 가중치로 새로운 모델을 초기화  

<br/>

### 4.3 Model Serving
- 모델이 훈련되고 검증이 끝나면 모델 서버에 모델을 적재한다. 

- 각 요청마다 서버는 앱 검색(Retrieval) 시스템으로부터 앱 후보셋을 받고, 사용자 변수를 사용하여 후보 앱에 점수를 매긴다. 

- 앱은 가장 높은 점수부터 가장 낮은 점수까지 순위가 매겨지며, 순위대로 사용자에게 노출한다. 

- 점수는 Wide & Deep 모델에 대한 순방향 추론을 실행하여 계산된다.

- 이때, 10ms 단위로 각 요청을 처리하기 위해 추론 단계에서 단일배치로 모든 후보앱에 점수를 매기는 대신, 멀티스레딩 병렬처리를 통해 미니배치를 병렬로 실행하여 성능을 최적화하였다. 

<br/>

## 5. Experiment Results
Wide & Deep 모델의 성능을 검증하기 위해 실험을 통해 두가지 측면에서 평가를 진행하였다.

### 5.1 App Acquisitions
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/148770435-66d8821b-738d-4665-a895-e7519d792eaf.PNG"></p>

- 실험은 3주간의 실시간 온라인 A/B 테스트로 진행되었는데, Wide-only, Deep-only, Wide & Deep 3가지 모델을 각각 랜덤으로 선택한 전체의 1%에 해당하는 사용자에게 적용하였다. 

- 실험 결과, Wide & Deep 모델이 Wide-only 모델(대조군)에 비해서는 3.9% 더 높은 앱 설치율을 보였고, Deep-only 모델에 비해서는 1% 더 높은 앱 설치율을 보였다. 

- 오프라인 테스트 셋을 통해 얻은 오프라인 AUC의 경우에도 Wide & Deep 모델이 가장 우수한 성능을 보였지만, 온라인 트래픽 만큼의 큰 영향을 주지 못했다. 

  - 이는 오프라인 데이터 셋에서는 노출과 레이블이 고정되어 있지만, 온라인에서는 generalization과 memorization을 함께 고려하여 탐색적으로 추천해볼 수 있고 그에 따라 사용자 응답으로 학습이 가능하다는 점이 이유 중 하나일 것이라고 한다. 

<br/>

### 5.2 Serving Performance
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/148770437-0a9bda40-768f-4f78-b7be-29fb76479570.PNG"></p>

- 상용 모바일 앱 스토어는 트래픽이 많기 때문에, 짧은 대기 시간 내 처리량을 높게 유지하며 서비스하기에 어려움이 많다. 

- 본 실험에서 멀티쓰레드를 구현하고 각각의 배치를 더 작은 사이즈로 나누었는데, 이는 client-side에서 지연시간을 14ms로 현저하게 감소시켰다. (31ms -> 14ms)

<br/>

## 6. Conclusion
- 추천시스템에서 Memorization과 Generalization은 모두 중요하다.

- Wide 선형 모델은 sparse 변수 간 상호작용을 cross-product feature transformation을 통해 효과적으로 memorize 할 수 있고, Deep neural network는 이전에 나타나지 않은 변수 긴 상호작용을 저차원 임베딩을 통해 일반화 시킬 수 있다. 

- 본 논문에서는, Wide와 Deep 모델 각각의 장점을 합치기 위해 Wide & Deep 학습 프레임워크를 제시하였다. 

- Wide & Deep 모델을 대규모 상업용 앱 스토어인 구글 플레이에서 구축하고 평가하였고, 온라인 실험 결과에 따르면 Wide만 또는 Deep만 사용한 모델보다 눈에 띄게 높은 앱 설치율을 보여주었다. 
