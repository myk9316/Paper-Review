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

<br/>

## 3. Candidate Generation
-

<br/>

### 3.1 Recommendation as Classification
- 

<br/>

#### Efficient Extreme Multiclass
-

<br/>

### 3.2 Model Architecture
- 

<br/>

### 3.3 Heterogeneous Signals
-

<br/>

#### "Example Age" Feature
-

<br/>

### 3.4 Label and Context Seletion
-

<br/>

### 3.5 Experiments with Features and Depth
-

<br/>

## 4. Ranking
-

<br/>

### 4.1 Feature Representation
-

<br/>

#### Feature Engineering
-

<br/>

#### Embedding Categorical Features
-

<br/>

#### Normalizing Continuous Features
-

<br/>

### 4.2 Modeling Expected Watch Time
-

<br/>

### 4.3 Experiments with Hidden Layers
-

<br/>

## 5. Conclusions
-

<br/>


