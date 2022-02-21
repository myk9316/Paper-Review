# Neural Collaborative Filtering
본 논문은 2017년에 발표된 논문으로,

<br/>

## Abstract
In recent years, deep neural networks have yielded immense success on speech recognition, computer vision and natural language processing. However, the exploration of deep neural networks on recommender systems has received relatively less scrutiny. In this work, we strive to develop techniques based on neural networks to tackle the key problem in recommendation — collaborative filtering — on the basis of implicit feedback. 
Although some recent work has employed deep learning for recommendation, they primarily used it to model auxiliary information, such as textual descriptions of items and acoustic features of musics. When it comes to model the key factor in collaborative filtering — the interaction between user and item features, they still resorted to matrix factorization and applied an inner product on the latent features of users and items. 
By replacing the inner product with a neural architecture that can learn an arbitrary function from data, we present a general framework named NCF, short for Neural networkbased Collaborative Filtering. NCF is generic and can express and generalize matrix factorization under its framework. To supercharge NCF modelling with non-linearities, we propose to leverage a multi-layer perceptron to learn the user–item interaction function. Extensive experiments on two real-world datasets show significant improvements of our proposed NCF framework over the state-of-the-art methods. Empirical evidence shows that using deeper layers of neural
networks offers better recommendation performance. 

<br/>

## 1. Introduction

<br/>


## 2. Preliminaries

<br/>


### 2.1 Learning from Implicit Data

<br/>

### 2.2 Matrix Factorization

<br/>

## 3. Neural Collaborative Filtering

<br/>

### 3.1 General Framework

<br/>

#### 3.1.1 Learning NCF

<br/>

### 3.2 Generalized Matrix Factorization (GMF)

<br/>

### 3.3 Multi-Layer Perceptron (MLP)

<br/>

### 3.4 Fusion of GMF and MLP

<br/>


#### 3.4.1 Pre-training

<br/>

## 4. Experiments

<br/>

### 4.1 Experimental Settings

<br/>

### 4.2 Performance Comparison (RQ1)

<br/>

#### 4.2.1 Utility of Pre-training

<br/>

### 4.3 Log Loss with Negative Sampling (RQ2)

<br/>

### 4.4 Is Deep Learning Helpful? (RQ3)

<br/>

## 5. Related Work

<br/>

## 6. Conclusion and Future WOrk

<br/>
