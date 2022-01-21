# BPR: Bayesian Personalized Ranking from Implicit Feedback

## Abstract
Item recommendation is the task of predicting a personalized ranking on a set of items (e.g. websites, movies, products). In this paper, we investigate the most common scenario with implicit feedback (e.g. clicks, purchases). There are many methods for item recommendation from implicit feedback like matrix factorization (MF) or adaptive knearest- neighbor (kNN). Even though these methods are designed for the item prediction task of personalized ranking, none of them is directly optimized for ranking. In this paper we present a generic optimization criterion BPR-Opt for personalized ranking that is the maximum posterior estimator derived from a Bayesian analysis of the problem. We also provide a generic learning algorithm for optimizing models with respect to BPR-Opt. The learning method is based on stochastic gradient descent with bootstrap sampling. We show how to apply our method to two state-of-the-art recommender models: matrix factorization and adaptive kNN. Our experiments indicate that for the task of personalized ranking our optimization method outperforms the standard learning techniques for MF and kNN. The results show the importance of optimizing models for the right criterion.

<br/>

## 1. Introduction
- 

<br/>


## 2. Related Work
- 

<br/>

## 3. Personalized Ranking
- 

<br/>

### 3.1 Formalization
- 

<br/>

### 3.2 Analysis of the problem setting
- 

<br/>


## 4. Bayesian Personalized Ranking (BPR)
- 

<br/>

### 4.1 BPR Optimization Criterion
- 

<br/>

#### 4.1.1 Analogies to AUC optimization
- 

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
