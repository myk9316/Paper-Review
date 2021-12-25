# BERT 기반 감성분석을 이용한 추천시스템

## Recommender System
본 논문은 2021년 지능정보연구(Journal of intelligence and information systems)에 실린 논문이다. 해당 논문은 사용자의 평점과 아이템 항목만을 사용하여 추천하는 기존의 전통적인 추천시스템 알고리즘의 한계를 지적하며, BERT 기반 감성분석을 이용하여 어떻게 리뷰데이터를 활용하여 성능을 개선함과 동시에 분석과 추천 과정을 단순화 하였는지 살펴보겠다.

## Abstract
추천시스템은 사용자의 기호를 파악하여 물품 구매 결정을 도와주는 역할을 할 뿐만 아니라, 비즈니스 전략의 관점에 서도 중요한 역할을 하기에 많은 기업과 기관에서 관심을 갖고 있다. 최근에는 다양한 추천시스템 연구 중에서도 NLP와 딥러닝 등을 결합한 하이브리드 추천시스템 연구가 증가하고 있다. NLP를 이용한 감성분석은 사용자 리뷰 데이터가 증가함에 따라 2000년대 중반부터 활용되기 시작하였지만, 기계학습 기반 텍스트 분류를 통해서는 텍스트의 특성을 완전히 고려하기 어렵기 때문에 리뷰의 정보를 식별하기 어려운 단점을 갖고 있다. 본 연구에서는 기계학습의 단점을 보완하기 위하여 BERT 기반 감성분석을 활용한 추천시스템을 제안하고자 한다. 비교 모형은 Naïve-CF(collaborative filtering), SVD(singular value decomposition)-CF, MF(matrix factorization)-CF, BPR-MF(Bayesian personalized ranking matrix factorization)-CF, LSTM, CNN-LSTM, GRU(Gated Recurrent Units)를 기반으로 하는 추천 모형이며, 실제 데이터에 대한 분석 결과, BERT를 기반으로 하는 추천시스템의 성과가 가장 우수한 것으로 나타났다. 이 논문을 읽어서 무엇을 배울 수 있는가?


## Introduction

- 기존의 전통적인 추천시스템 알고리즘 주로 명시적 데이터인 사용자의 평점과 아이템 항목만을 사용하여 개인화 콘텐츠와 지능형 콘텐츠 필터에 대한 한계점이 존재하여 추천의 정확도가 떨어지는 문제가 존재한다. 

- 선행 연구에서 사용자 평점 뿐만 아니라 사용자의 리뷰 데이터와 같은 기타의 감성 정보를 활용하면 추천의 성능이 개선될 수 있을 것이라고 한다.  

- 따라서, 본 연구에서는 BERT의 감성분석을 이용하여 소비자 리뷰의 문맥정보까지 반영할 수 있는 새로운 추천시스템을 제안하고자 한다. 

- 또한, 선행 연구들에서는 고차원의 벡터를 줄이는 차원 축소에 NLP를 적용해왔으나, 본 연구에서는 차원 축소를 통해 발생하는 데이터 손실을 최소화 하기 위해 벡터의 차원을 줄이지 않고 딥러닝을 이용하여 소비자의 관심사와 평점을 그대로 활용하여 분석을 진행하고자 한다. 

## Model Architecture
**BERT : Bidirectional Encoder Representations from Transformers**

![enter image description here](https://user-images.githubusercontent.com/79245484/147387211-f51e77c6-8e35-4ea8-b884-73d3077b159e.PNG)

### BERT를 이용한 감성 분석

 1.  3가지의 입력 임베딩(Token, Segment, Position)을 취합하여 하나의 임베딩 값으로 만들고, 이를 input으로 사용한다. 
 
 3. Encoder에 input을 넣을 때 해당 문장의 30%의 token을 임의로 Mask 처리하고, 해당 token을 예측하여 학습을 진행한다. 
 
 5.  사전 학습 모델은 BERT multilingual base model을 활용하여 사용자 리뷰와 평점을 벡터로 입력 받아 BERT 감성 분석을 할 수 있도록 구현한다. 


### BERT를 활용한 추천 시스템

 1. BERT 모델의 장점은 포지션 임베딩을 이용하여 문장에서 한 쌍의 단어 관계를 정의하여 진행 한다는 점인데,  Scaled dot-product attention 방법을 사용하여 한 쌍의 단어 관계를 나타낸다.
 - 아래 식의 나오는 것처럼, Query(Q)와 Key(K)의 벡터 간 유사도를 행렬곱 연산을 통해 구한 후에,  $\sqrt{d_k}$ 를 scaling factor로 사용하여 나눈 뒤 softmax를 적용하고 거기에 Value(V)와 다시 행렬곱 연산을 한다. 


![enter image description here](https://user-images.githubusercontent.com/79245484/147388184-ec512ff5-8d48-4f79-be8c-90698c367fe0.PNG)

   이를 통해 만들어지는 예시는 아래와 같다. 
   
![enter image description here](https://user-images.githubusercontent.com/79245484/147388577-776709dc-6f5d-49fd-83a7-9bc4e2bd5f0e.PNG)


 3. 다음은 Bi-LSTM과 Attention layer를 진행한다. 
 - Gradient vanishing problem을 해결하고 문맥 정보를 활용하기 위해 LSTM을 사용하였고, 감성분석의 입력 데이터에 대해 LSTM을 훈련시키기 위해 양방향으로 LSTM을 적용할 수 있는 Bi-LSTM을 사용한다. 또한, Bi-LSTM을 사용하면 트랜스포머 계층에서 숨겨진 token을 연결하여 전체 모델을 다시 미세 조정할 수 있다. 


	![enter image description here](https://user-images.githubusercontent.com/79245484/147388578-30d9c740-a893-4bae-8057-56a0cb1088ca.PNG)


- 하나의 문장 정보를 문장 끝까지 입출력 할 수 있도록 하여 첫 단어가 멀리 있는 단어와도 상관관계를 가질 수 있도록 Attention layer를 사용한다. 


 6. 다음은, Attention layer 값을 하나로 줄이기 위해 Max Pooling을 사용하며,  Max Pooling은 계산량을 줄임으로써 과적합을 방지할 수 있다. 
 
 8. 마지막으로, Output layer에서 softmax 활성화 함수를 사용하여 분류된 문장 쌍을 이용하고, 각 평점 레이블의 확률 값에 따라 출력 값을 산출한다. 


## Experiments
### 데이터 셋과 전처리
- 아마존 미국 사이트의 음식 분야의 약 32만 개의 리뷰 데이터를 크롤링하였다.

- 그 중 helpfulness가 5개 이하의 데이터는 제거하고, 평점과 부합하지 않는 리뷰들이 다수 존재하여 평점과 리뷰를 매칭했다.

-  매칭 된 리뷰는 총 42,283개이며, 최종 데이터 셋을 Training Set 80%(33,286개), Test Set 20%(8,457개)로 구성하였으며 데이터 셋에 대한 상세정보는 다음과 같다. ![enter image description here](https://user-images.githubusercontent.com/79245484/147388935-56c0748a-ad58-4d87-9f2f-47a099d1ad70.PNG)

### 감성분석 결과
- BERT를 통해 분석된 감성분석의 결과를 비교하기 위해 Accuracy, Precision, Recall, F1-score를 사용하였다. 

- BERT는 512byte 제약이 있고 리뷰 길이가 길 경우에 시간이 오래 걸리거나 무한 루프에 빠질 위험이 있다. 따라서, 본 논문에서 사용되는 리뷰 길이에 대한 설정은 BERT 학습에서 config는 base BERT로 사용하여 90%를 128byte의 built-in 데이터 구조로 설정한 후, 10%는 512byte로 학습하였다. 

- Config, batch size(=64), thread(=4)로 설정하고, Wordpiece 알고리즘을 이용하여 일부 단어에 대한 전처리와 마스킹을 동시에 진행하였다. 

- 마스킹 처리 후 감성분석과 BERT 모델을 결합하기 위해 파이토치의 tensor를 이용하였고, 감성분석의 가중치에 패널티를 부여하기 위해서 Adam 함수를 이용하였다. 

- Adam 함수를 이용한 후 5000번의 반복 시행을 거쳐 생성된 데이터셋에 로지스틱 회귀분석을 적용한 결과는 다음과 같다. ![enter image description here](https://user-images.githubusercontent.com/79245484/147388937-be0096fd-2078-488c-9566-214ed58ae820.PNG)

### 추천 성능 평가
- RMSE를 상품 추천의 평가지표로 사용하였다. 

- 기존 추천시스템 알고리즘과 딥러닝 모델들에 비해 BERT가 RMSE와 MSE 측면에서 가장 우수한 추천 성과를 보여주었으며, 결과는 다음과 같다. ![enter image description here](https://user-images.githubusercontent.com/79245484/147389086-7a31102d-e8a6-48b9-ab0c-6e4b5b821aa0.PNG)


## Conclusion
- 본 연구에서 제안한 BERT 기반의 감성분석과 추천모델이 비교모형에 비해 높은 성능을 보였다.  

- 전통적인 추천 알고리즘에서는 사용자의 평점과 아이템의 항목만을 사용하여 추천을 하는데, 이러한 정량적인 정보에만 국한하면 추천의 정확도가 떨어진다는 문제가 존재한다. 이러한 문제를 해결하기 위해 사용자의 리뷰데이터를 문맥을 고려하여 감성분석을 진행하고, 이를 추천시스템에 반영한 것이 우수한 추천 성능을 이끌어냈다고 생각된다. 

- 또한, BERT를 기반으로 한 감성 분석과 추천을 동시에 진행하여, 감성 분석의 결과를 추천시스템에 직접적으로 반영하여 분석과 추천과정이 단순화 시킨 것이 본 논문의 contribution이라고 생각된다. 


## Limitation
- BERT의 연산과정이 난해하여 이에 대한 이해와 모델링 과정에서 상당한 시간이 소요된다.

- 딥러닝의 특성 상 결과를 해석하는 설명력에서의 한계점이 있다.

- 콜드스타트 문제를 해결할 수 있는 방안은 고려하지 않았다.

- 추천의 성능 평가에 있어서 추천시스템에서 많이 활용되고 있는 Top-N까지 고려하지 못한다.

## Advice
- BERT를 사용한 방법을 설명할 때 좀 더 체계적으로 설명해주면 좋았을 것 같다. 본 논문을 읽으면서 BERT에 대해 설명한 다른 글들도 찾아보았는데, 본 논문에 비해 더 잘 이해가 갔던 것 같다. (나의 이해력 문제일수도...)

- 본 연구에서는 음식 데이터만 가지고 연구를 진행해서 높은 추천 성능을 얻었는데, 다른 도메인의 데이터를 사용하여 모델을 평가하였을 때는 좋지 않은 성능이 나올 수도 있지 않을까? 라는 생각이 든다. (물론 다른 도메인에서도 성능이 더 좋게 나올 수도 있다) 
	- 음식 데이터 이외의 다른 도메인의 데이터도 수집하여 모델 평가를 해보면 좋을 것 같다. 특히, GRU와의 성능 차이가 크지 않았다고 하는데, 다른 도메인의 데이터를 사용해 평가함으로써 BERT가 항상 GRU보다 성능이 우월한지도 궁금하다. 
