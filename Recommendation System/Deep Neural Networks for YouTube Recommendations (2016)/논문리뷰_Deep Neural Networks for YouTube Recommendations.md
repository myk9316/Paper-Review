# Deep Neural Networks for YouTube Recommendations
본 논문은 2016년에 발표된 논문으로, 구글에서 딥러닝을 적용해 구성한 유투브 추천 알고리즘에 대해 설명한다. 

<br/>

## Abstract
YouTube represents one of the largest scale and most sophisticated industrial recommendation systems in existence. In this paper, we describe the system at a high level and focus on the dramatic performance improvements brought by **deep learning**. The paper is split according to the classic two-stage information retrieval dichotomy: first, we detail a **deep candidate generation model** and then describe a separate **deep ranking model**. We also provide practical lessons and insights derived from designing, iterating and maintaining a massive recommendation system with enormous userfacing impact.

## 1. Introduction
- 유투브의 추천은 세 가지 주요 challege가 존재한다.
  - Scale : 기존에 존재하던 추천 알고리즘들이 작은 규모의 데이터에는 잘 작동하지만, 데이터가 굉장히 많은 유튜브에는 잘 작동되지 않은 경우가 많다. 
  
  - Freshness : 새로운 비디오가 굉장히 자주 그리고 많이 업로드 되기 때문에 추천에 바로 반영되어야 하고, 이때 사용자의 최근 행동도 고려되어야 한다. 
  
  - Noise : 사용자의 과거 행동은 sparsity 하고 다양한 관찰되지 않은 외부 요인들이 존재하기 때문에 예측하기가 어렵다. 또한, Explicit feedback을 구하기 어렵기 때문에 implicit feedback이 자주 사용되며, meta data 구축이 어렵다. 

<br/>

## 2. System OverView
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/156601223-17e20a7b-cde7-42f6-9a47-42cbc403aac2.PNG" width="50%" height="50%"></p>

- Candidate generation
  - 사용자의 활동 기록을 input으로 해서 Collaborative filtering을 통해 사용자의 성향이 반영된 적합한 수백개의 후보 영상들을 output으로 내보낸다.
  - 사용자 간의 유사도는 비디오 시청 id, 검색어, 그리고 인구통계학 정보와 같은 feature들을 통해 표현된다. 


- Ranking
  - 비디오와 사용자를 설명하는 다양한 피처들을 사용하여 objective function에 따라 각 비디오에 점수를 매긴다. 
  - Score가 높은 비디오가 사용자에게 제공된다. 

- 이러한 두 단계를 거치며 매우 큰 비디오의 corpus로 부터 개인화된 적은 수의 비디오가 사용자에게 추천된다. 
- 개발 과정에서는 offline metrics(prcision, recall, ranking loss, etc.)를 구축하여 시스템의 성능을 향상하지만, 최종적으로 실제 환경에서의 효율성을 검증하기 위해서는 라이브 실험을 통한 A/B 테스트를 사용한다. 

<br/>

## 3. Candidate Generation
- 이 단계에서는 엄청난 규모의 유튜브 corpus에서 사용자와 관련된 몇 백개의 영상으로 후보군을 추출한다.
- 네트워크를 구성하는 아이디어는 이전의 Matrix Factorization의 많은 영향을 받았으며, 본 논문에서 제안하는 이러한 방법은 factorization의 non-linear generalization으로 볼 수 있다. 
- Fully connected ReLu

<br/>

### 3.1 Recommendation as Classification
- 본 논문에서 제안하는 추천은 엄청나게 많은 클래스(extreme multiclass classification)가 있는 분류 문제로 정의한다. 
  - 즉, 사용자(U)와 Context(C)를 기반으로 Corpus(V)에서 수백만 개의 비디오(i) 중 특정 시간(t)에 시청한 특정 비디오(<img src="https://latex.codecogs.com/svg.image?w_{t}" title="w_{t}" />)를 분류하는 문제

- <p align="center"><img src="https://user-images.githubusercontent.com/79245484/156601226-cd43c004-853c-48cf-a7c7-6b2cb74a1831.PNG" width="30%" height="30%"></p>

  - <img src="https://latex.codecogs.com/svg.image?u&space;\epsilon&space;&space;\mathbb{R}^{N}" title="u \epsilon \mathbb{R}^{N}" /> : 고차원적 사용자 Embedding (사용자 정보와 Context 정보를 조합)
  - <img src="https://latex.codecogs.com/svg.image?v_{j}&space;\epsilon&space;&space;\mathbb{R}^{N}" title="v_{j} \epsilon \mathbb{R}^{N}" /> : 각 후보 비디오의 Embedding

- 해당 세팅에서 임베딩은 단순하게 개별 비디오, 사용자 등의 sparse 한 entities를 dense vector로 mapping 하는 것이다. 

- deep neural network의 task는 사용자 기록 및 context의 함수로서 사용자 임베딩 u를 학습하는 것을 바탕으로 softmax 분류기를 통해서 각 비디오 시청 확률을 예측하는 것이다.

- 이러한 학습 과정에서 explicit feedback(사용자가 누른 좋아요/싫어요 등..)은 사용하지 않고, implicit feedback(사용자가 비디오를 끝까지 시청했는지/안했는지)을 사용하였다. 


<br/>

#### Efficient Extreme Multiclass
- 학습 단계 (training)
  - 엄청나게 많은 클래스를 분류하는 모델을 효율적으로 학습시키기 위해 sample negative classes (negative sampling) 기법을 적용한다. 
    - 수천개의 샘플만 뽑아서 샘플링 된 것을 학습
  - 이러한 방식을 적용한 이유로는 클래스가 너무 많을때 가능한 모든 클래스에 대해서 내적을 수행하기 때문에 Softmax 연산 cost가 기하급수적으로 늘어나기 때문이다. 

- 실시간 추천단계 (serving time)
  - 사용자에게 top N개의 비디오를 추천하기 위한 점수를 계산할 때는 dot product space에서 최근접 이웃(nearest neighbor)을 search하는 과정을 거친다. 
  - A/B 테스트 결과, nearest neighbor 알고리즘 간의 성능 차이는 없다고 한다.

<br/>

### 3.2 Model Architecture
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/156601232-6354d9d8-8034-435b-ba2d-fe0ca89dc207.PNG" width="50%" height="50%"></p>

- Embedding : Video Embedding + Search Token Embedding + Demographic feature

- Embedded video wathces --> watch vector
  - 고정된 vocabulary 안에서 각 비디오에 대한 고차원의 임베딩을 학습하고, 이러한 임베딩을 feedforward neural network에 feed 한다. 
  
  - 사용자의 시청 기록은 희소한 비디오 ID의 가변 길이 sequence에 의해 표현되며, Dense vector로 임베딩된다. 이때, embedding vectors를 평균내서 사용하는 것이 성능이 가장 좋다. 

<br/>

### 3.3 Heterogeneous Signals
- Matrix factorization의 deep neural network 사용의 이점은 연속형 변수와 범주형 변수를 모델에 쉽게 추가할 수 있다는 점이다. 

- Embedded search tokens --> search vector
  - 검색 내역도 시청 내역과 비슷하게 처리되는데, unigram/bigram 단위로 임베딩하여 평균 낸 것이 요약된 dense search history로 표현된다. 

- Geographic embedding
  - 그 외에 사용자의 지리적 정보 및 사용 기기 등의 인구통계학적 정보들도 임베딩하고 이를 concatenate 한다. 
  - 이때 사용자의 성별, 나이 등의 값은 [0, 1]로 normalized 되어 실수 값으로 입력된다. 

<br/>

#### "Example Age" Feature
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/156601230-eae83324-21f0-4037-a28f-c0a70256e2ae.PNG" width="40%" height="40%"></p>

- 매초마다 대량의 새로운 영상들이 업로드 되는데, 이러한 새로운 비디오를 잘 추천하는 것이 중요하다. 왜냐하면, 사용자들이 새로운 컨텐츠를 선호하는 것을 지속적으로 관찰해왔기 때문이다. 

- 머신러닝 시스템은 미래를 예측하도록 훈련이 되어있기 때문에 과거 아이템에 대한 경향을 보여준다. 즉, 단순하게 오래된 아이템들이 더 많은 추천을 받게 된다. 

- 이를 해결하게 위해 아이템의 '나이'를 피처로 추가해주었으며, 위의 그래프를 보면 나이를 추가함으로 성능이 향상되었다. 즉, 나이를 추가함으로써 업로드 직후에 많은 사람들이 시청하는 경향을 보여준다. 

<br/>

### 3.4 Label and Context Seletion
- 추천시스템은 surrogate problem(평점을 예측하는 모델이 좋은 추천을 할 수 있는지)을 자주 마주한다. 
  - surrogate problem: 개발한 추천 엔진에 대한 평가는 직접 서빙하여 사용자로부터 피드백을 받는 것이지만 항상 이러한 형태의 평가가 불가능함으로, RMSE 혹은 MAP와 같은 성능 지표를 활용하여 모델을 평가하는 과정

- 학습데이터는 유튜브 추천시스템의 결과로부터만 생성되는 것보다 다른 사이트의 유입까지 모두 포함하는 것이 좋다. 
  - 그렇지 않으면, 데이터가 유튜브 추천시스템의 의한 결과이기 때문에 bias가 생기게 되어, 새로운 아이템을 탐색하기가 어렵다. (biased toward exploitation, rather than exploration) 

- 또한, loss function에서 모든 사용자의 가중치를 동일하게 유지하기 위해 사용자마다 학습 데이터의 양을 고정하는 것이 좋다. 
  - 매우 활동적인 사용자의 영향이 dominate 하는 것을 방지한다. 

- 추천 결과나 검색 결과를 즉시 사용하지 않는다. 왜냐하면, 방금 검색한 내용이 계속 표시된다. 

- 유튜브 영상 감상 패턴은 매우 비대칭적이다. 즉, 영상을 규칙적으로 보지 않는다 (ex. 에피소드 순서대로, 유명한 것부터 감상, 게임 영상을 볼때는 게임 영상만 감상). 따라서, 추천 결과를 예측함에 있어 무작위로 선택된 아이템을 예측하는 것보다, 사용자의 다음 시청 영상을 예측하는 것이 성능이 더 좋았다. 
  - 아래 그림의 (b)에서는 특정 시점의 아이템을 예측하기 위해 특정 시점 이전의 데이터(watch history)로만 학습데이터를 구축한다. 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/156601235-c1f6914f-f2b1-4d5a-bdb6-fc86292995a4.PNG" width="80%" height="80%"></p>

<br/>

### 3.5 Experiments with Features and Depth
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/156601238-1749e686-aa6a-4e72-bf92-f1dffd7cc144.PNG" width="40%" height="40%"></p>

- 비디오/검색어 각각 100만개를 256차원의 임베딩으로 변환
- 인당 최대 최근 시청 50개, 검색 50개
- feature를 추가하고 depth가 깊어질수록 성능이 좋아졌다. 

<br/>

## 4. Ranking
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/156601241-53cd64c7-b11c-4eaa-86fc-651ebb7d6456.PNG" width="80%" height="80%"></p>

- 랭킹 모델의 주 역할은 impression data를 사용해서 후보 아이템을 특성화하고 조정해서 특정 사용자 인터페이스에 대한 후보 예측을 수행한다. 
- 이 때, 비디오와 사용자의 관계를 설명해주는 더 많은 features를 이용할 수 있다. (위에선 몇백개만 봤기 때문에 추가적으로 놓친 것을 파악하기 위해)
- 또한, 랭킹은 직접 비교하기가 어려운 여러 candidate 들을 앙상블하는데도 중요하다. 
- Candidate generation 모델과 구조는 유사하며(deep neural network), 각 아이템마다 점수를 할당하여 정렬 후 사용자에게 return 한다. 

<br/>

### 4.1 Feature Representation
- Feature는 데이터의 형태에 따라서 categorical과 continuous/ordinal 로 구분할 수 있다. 

  - Categorical feature
    - univalent : 값을 한가지만 가지는 feature / ex. scoring하는 비디오의 ID 
    - multivalent : 값을 여러개 가지는 feature / ex. 사용자가 지금까지 봤던 비디오의 ID 목록

- 또한, Feature는 데이터 의미에 따라서 Query와 impression으로 구분할 수 있다. 
  - Query : 사용자/context의 특성을 설명  , 한번만 computed됨
  - Impression : 아이템의 특성을 설명  , 아이템이 점수를 부여 받을 때마다 computed됨

<br/>

#### Feature Engineering
- 수백개의 feature를 사용한다. 

- 딥러닝이 피처엔지니어링에 대한 수고스러움을 줄어주긴하지만, feedfoward neural network에 넣기 위해서는 raw data에 대한 전처리가 여전히 필요하다. 

- Main challenge는 사용자 행동에 대한 sequence를 표현하는 것과 이러한 행동들이 점수화된 비디오 impression과 어떻게 관련 있는지를 표현하는 것이다. 

- 이때 가장 중요한 signal은 아이템 자체 및 랭킹 광고 경험과 일치하는 유사한 아이템에 대한 사용자의 이전 상호작용이다. 
  - 예를 들어, 해당 채널에서 시청한 비디오의 개수, 해당 토픽과 관련된 비디오를 마지막으로 시청한 시점 등과 같은 아이템과 관련된 사용자의 과거 행동의 continuous feature는 매우 효과적이다. 

- 또한 왜 이 비디오가 추천되었는지 정보를 활용하고, 추천되었는데 보지 않은 비디오는 순위를 깎는다. 

<br/>

#### Embedding Categorical Features
- Categorical feature들은 dense embedding으로 맵핑되서 neural network에 input으로 입력된다.
- 가질 수 있는 값이 굉장히 많은 categorical feature에 경우에는 click 빈도수를 기반으로 정렬 후 Top N 개만 사용되고, 나머지는 Out-of-vocabulary로 취급되어 zero embedding으로 변환된다.
- 값을 여러개 가질 수 있는 multivalent categorical feature들은 평균을 낸 다음에 입력이 된다. 

<br/>

#### Normalizing Continuous Features
-

<br/>

### 4.2 Modeling Expected Watch Time
-

<br/>

### 4.3 Experiments with Hidden Layers
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/156601245-1d7bf46a-30e7-4c8c-82d6-09b7ef7a335c.PNG" width="50%" height="50%"></p>

-

<br/>

## 5. Conclusions
-

<br/>


