# Wide & Deep Learning for Recommeder Systems
본 논문은 2006년 구글에서 발표한 wide and deep 추천랭킹 알고리즘에 관한 논문이다. 이 알고리즘은 구글플레이에서 앱 추천에 실제로 적용된 알고리즘으로, 선형모델과 신경말 모델을 함께 적용하여 Memorization과 Generalization 둘다를 얻어 추천랭킹을 개선하는 것이 목표이다. 

<br/>

## ABSTRCT
Generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification problems with sparse inputs. Memorization of feature interactions through a wide set of cross-product feature transformations are effective and interpretable, while generalization requires more feature engineering effort. With less feature engineering, deep neural networks can generalize better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features. However, deep neural networks with embeddings can over-generalize and recommend less relevant items when the user-item interactions are sparse and high-rank. In this paper, we present Wide & Deep learning—jointly trained wide linear models and deep neural networks—to combine the benefits of memorization and generalization for recommender systems. We productionized and evaluated the system on Google Play, a commercial mobile app store with over one billion active users and over one million apps. Online experiment results show that Wide & Deep significantly increased app acquisitions compared with wide-only and deep-only models. We have also open-sourced our implementation in TensorFlow.

<br/>

## 1. Introduction
- 

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


