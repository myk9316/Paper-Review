# BERT 기반 감성분석을 이용한 추천시스템

## Recommender System
본 논문은 2021년 지능정보연구(Journal of intelligence and information systems)에 실린 논문이다. 해당 논문은 사용자의 평점과 아이템 항목만을 사용하여 추천하는 기존의 전통적인 추천시스템 알고리즘의 한계를 지적하며, BERT 기반 감성분석을 이용하여 어떻게 리뷰데이터를 활용하여 성능을 개선함과 동시에 분석과 추천 과정을 단순화 하였는지 살펴보겠다.

## Abstract
추천시스템은 사용자의 기호를 파악하여 물품 구매 결정을 도와주는 역할을 할 뿐만 아니라, 비즈니스 전략의 관점에 서도 중요한 역할을 하기에 많은 기업과 기관에서 관심을 갖고 있다. 최근에는 다양한 추천시스템 연구 중에서도 NLP와 딥러닝 등을 결합한 하이브리드 추천시스템 연구가 증가하고 있다. NLP를 이용한 감성분석은 사용자 리뷰 데이터가 증가함에 따라 2000년대 중반부터 활용되기 시작하였지만, 기계학습 기반 텍스트 분류를 통해서는 텍스트의 특성을 완전히 고려하기 어렵기 때문에 리뷰의 정보를 식별하기 어려운 단점을 갖고 있다. 본 연구에서는 기계학습의 단점을 보완하기 위하여 BERT 기반 감성분석을 활용한 추천시스템을 제안하고자 한다. 비교 모형은 Naïve-CF(collaborative filtering), SVD(singular value decomposition)-CF, MF(matrix factorization)-CF, BPR-MF(Bayesian personalized ranking matrix factorization)-CF, LSTM, CNN-LSTM, GRU(Gated Recurrent Units)를 기반으로 하는 추천 모형이며, 실제 데이터에 대한 분석 결과, BERT를 기반으로 하는 추천시스템의 성과가 가장 우수한 것으로 나타났다. 이 논문을 읽어서 무엇을 배울 수 있는가?


## Introduction

- 기존의 전통적인 추천시스템 알고리즘 주로 명시적 데이터인 사용자의 평점과 아이템 항목만을 사용하여 개인화 콘텐츠와 지능형 콘텐츠 필터에 대한 한계점이 존재하여 추천의 정확도가 떨어지는 문제가 존재한다. 
- 선행 연구에서 사용자 평점 뿐만 아니라 사용자의 리뷰 데이터와 같은 기타의 감성 정보를 활용하면 추천의 성능이 개선될 수 있을 것이라고 하였다.  
- 따라서, 본 연구에서는 BERT의 감성분석을 이용하여 소비자 리뷰의 문맥정보까지 반영할 수 있는 새로운 추천시스템을 제안하고자 한다. 
- 또한, 선행 연구들에서는 고차원의 벡터를 줄이는 차원 축소에 NLP를 적용해왔으나, 본 연구에서는 차원 축소를 통해 발생하는 데이터 손실을 최소화 하기 위해 벡터의 차원을 줄이지 않고 딥러닝을 이용하여 소비자의 관심사와 평점을 그대로 활용하여 분석을 진행하고자 한다. 

## Model Architecture
**BERT : Bidirectional Encoder Representations from Transformers**

![enter image description here](https://user-images.githubusercontent.com/79245484/147387211-f51e77c6-8e35-4ea8-b884-73d3077b159e.PNG)






## Experiments

## Conclusion
- 

### Limitation
- BERT의 연산과정이 난해하여 이에 대한 이해와 모델링 과정에서 상당한 시간이 소요됨
- 딥러닝의 특성 상 결과를 해석하는 설명력에서의 한계점이 있음
- 콜드스타트 문제를 해결할 수 있는 방안은 고려하지 않음
- 추천의 성능 평가에 있어서 추천시스템에서 많이 활용되고 있는 Top-N까지 고려하지 못함
### Advice

