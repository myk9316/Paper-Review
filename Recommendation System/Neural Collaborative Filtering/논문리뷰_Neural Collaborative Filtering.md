# Neural Collaborative Filtering
본 논문은 2017년에 '_the 26th international conference on world wide web_'에서 발표된 논문으로, 기존 Matrix Factorization 기반 모델에 Neural network를 결합한 추천시스템 모델을 제안한다. 

<br/>

## Abstract
In recent years, deep neural networks have yielded immense success on speech recognition, computer vision and natural language processing. However, the exploration of deep neural networks on recommender systems has received relatively less scrutiny. In this work, we strive to develop techniques based on neural networks to tackle the key problem in recommendation — collaborative filtering — on the basis of implicit feedback. 
Although some recent work has employed deep learning for recommendation, they primarily used it to model auxiliary information, such as textual descriptions of items and acoustic features of musics. When it comes to model the key factor in collaborative filtering — the interaction between user and item features, they still resorted to matrix factorization and applied an inner product on the latent features of users and items. 
By replacing the inner product with a neural architecture that can learn an arbitrary function from data, we present a general framework named NCF, short for Neural networkbased Collaborative Filtering. NCF is generic and can express and generalize matrix factorization under its framework. To supercharge NCF modelling with non-linearities, we propose to leverage a multi-layer perceptron to learn the user–item interaction function. Extensive experiments on two real-world datasets show significant improvements of our proposed NCF framework over the state-of-the-art methods. Empirical evidence shows that using deeper layers of neural
networks offers better recommendation performance.  

<br/>

## 1. Introduction
- Collaborative filtering에서 MF가 효과적인 방법이긴 하지만 linear한 방식이므로 사용자와 아이템간의 복잡한 관계를 표현하는데 한계가 존재한다. 
- 따라서 본 논문에서는 non-linear한 방식인 deep neural network(DNNs)의 활용을 제안하고, implicit feedback 데이터를 활용하여 모델의 검증을 진행한다. 

### Contribution
- Neural network에 기반한 Collaborative filtering 방식인 NCF를 제시한다.
- MF가 NCF의 특별한 케이스임을 보여주고, 고차원의 비선형 모델을 만들기 위해 다중레이어 퍼셉트론을 이용한다.
- 다양한 실험을 통해서 NCF의 효율성을 증명하고, CF 모델의 딥러닝적 접근이 가능함을 보여준다. 

<br/>

## 2. Preliminaries

### 2.1 Learning from Implicit Data
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/155169444-9260617e-ea85-414c-89c4-53eb519661fd.PNG" width="50%" height="50%"></p>

- Implicit data의 경우는 사용자 u와 아이템 i 간의 상호작용이 있을 경우에는 1, 없을 경우에는 0으로 표현한다. 
  - 이때, 1이라고 사용자 u가 아이템 i를 선호하는 것을 뜻하는 것은 아니다. 
  - 마찬가지로, 0이라고 사용자 u가 아이템 i를 비선호하는 것을 뜻하는 것은 아니다. 유저가 인지하지 못하여 상호작용이 없는 것일 수도 있다. 

- Implicit data의 문제는 사용자의 선호도를 분명하게 정의할 수 없다는 것이다. 
  - 사용자와 아이템 간의 상호작용이 발생하지 않은 경우 이를 missing data로 처리해야 하며, 부정적인 피드백은 항상 부족하다. 

- 따라서, 관찰되지 않은 항목의 점수의 예측은 다음과 같은 식으로 표현된다. <img src="https://user-images.githubusercontent.com/79245484/155175189-736142d1-c44b-4b3b-afba-c1d7463e2711.PNG" width="15%" height="15%"></p>
- 이때, 파라미터를 예측하기 위한 loss function으로는 pointwise loss와 pairwise loss를 사용할 수 있다.
  - pointwise loss: 회귀에 주로 사용되는 방법으로 실제값과 예측 값의 차이(오차)를 최소화한다. 
  - pairwise loss: 관측된 값이 관측되지 않은 값보다 큰 값을 가지도록 하기 위해 두 값의 마진을 최대화한다. 

- NCF는 interaction function _f_ 에 neural network를 적용하는 것으로 pointwise와 pairwise 둘다 이용할 수 있다. 

<br/>

### 2.2 Matrix Factorization
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/155169457-e269113c-37e7-4949-9e9e-07fede2cc9c4.PNG" width="50%" height="50%"></p>

- 사용자와 아이템의 inner product를 통해 interaction을 표현한다. 

### The Limiation of Matrix Factorization
- 본 논문에서 저자는 inner product가 MF의 표현력을 제한한다고 지적한다. 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/155169454-2632c4eb-391f-4f6d-b952-2c0064f1ab9f.PNG" width="40%" height="40%"></p>

- 위 그림에서, u1, u2, u3의 유사도에 따라서 그래프를 그리면 (b)와 같이 표현된다. 이때, u4라는 새로운 데이터에 대해 유사도를 측정하면, u1>u3>u2 순으로 유사하다. 하지만, (b)에서 보다시피 이를 표현할 수 있는 벡터를 그리는 것은 불가능하다. 

- 이러한 한계는 복잡한 사용자-아이템 간의 상호작용을 저차원의 latent space에서 단순하고 고정된 inner product로 표현하는데서 발생한다. 따라서, 본 논문에서는 DNN을 이용해 이러한 한계점을 해결하고자 한다. 

<br/>

## 3. Neural Collaborative Filtering
### 3.1 General Framework
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/155169460-8a74c716-0d81-4d51-97d8-3c26582c1d53.PNG" width="50%" height="50%"></p>

### Input Layer (Sparse)
- Input layer는 사용자 u와 아이템 i를 나타내는 feature vector <img src="https://latex.codecogs.com/svg.image?v_{u}^{U}" title="v_{u}^{U}" /> 와 <img src="https://latex.codecogs.com/svg.image?v_{i}^{I}" title="v_{i}^{I}" />로 구성되어 있으며, 각각의 벡터는 one-hot encoding으로 변환되어 sparse한 상태이다. 

### Embedding Layer
- Embedding layer에서는 sparse 한 input layer를 dense vector로 변환시키며, 일반적인 임베딩 방법과 동일하게 fully-connecte layer를 사용한다. 
- 이러한 임베딩 과정이 MF에서 잠재벡터의 역활을 한다고 볼 수 있다. 

### Neural CF Layers
- 이 단계에서는, 임베딩이 완료된 User latent vector와 Item latent vector를 concatenation한 벡터를 input으로 받아 여러 층의 신경망을 거치게 되는데, 이러한 다층 신경망 구조를 Neural CF Layers라고 한다. 
- Neural CF Layers를 통과하며 복잡한 비선형의 데이터 관계를 학습해서 score를 예측하게 된다. 

### Output Layer
- 마지막으로, output layer에서는 예측값을 구하게 되며, 실제값과 예측값 간의 pointwise loss를 최소화하는 방식으로 학습한다. 
- 여기서 예측값은, user u와 item i가 얼마나 관련있는지를 나타내며, <img src="https://latex.codecogs.com/svg.image?\phi_{out}&space;" title="\phi_{out} " />에 Logistic 또는 probit 함수를 활성화 함수로 사용하여 그 값은 0~1 사이가 된다. 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/155169464-07931100-2482-4533-9cc5-3a5b1beda8f1.PNG" width="50%" height="50%"></p>

<br/>

#### 3.1.1 Learning NCF
- 위와 같은 모델을 학습하기 위해서는 loss function을 정의해야하는데, 데이터들이 Gaussian distribution에서 왔다는 가정하에 squared loss를 다음과 같이 정의할 수도 있다. 
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/155169466-ed296791-4525-4e88-a089-7801319b9545.PNG" width="40%" height="40%"></p>

- 하지만, NCF는 binary 형태의 implicit data를 사용하여 학습하는 구조이기 때문에, bernoulli distribution을 가정하였다. 이에 따라, likelihood function은 다음과 같다.
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/155169474-ee75a519-868b-4ece-8202-5feea492bcd5.PNG" width="50%" height="50%"></p>

- 따라서, objective function로 사용하기 위한 negative logarithm of the likelihood는 다음과 같이 정의되며 (binary cross-entrophy loss와 동일), optimizaer로는 SGD를 사용한다. 모델은 L을 최소화하는 파라미터를 찾도록 학습한다. 
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/155169478-ff11ce92-3fa5-41c5-ad47-cb57e63d2057.PNG" width="50%" height="50%"></p>

<br/>

### 3.2 Generalized Matrix Factorization (GMF)
- 저자는 MF를 NCF의 특별한 케이스라고 말하며, GMF는 <img src="https://latex.codecogs.com/svg.image?a_{out}" title="a_{out}" />과 <img src="https://latex.codecogs.com/svg.image?h^{T}" title="h^{T}" />를 두어 MF를 일반화/확장화한 모델이다.

- 단순히 dot-product로 output을 예측한 MF와 달리 GMF에서는 다음과 같이 element-wise product를 수행하며 가충치를 적용한 후에 활성화 함수를 통과한다. 
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/155169485-2b2c19ad-2f6d-496f-82eb-faad74dc2f84.PNG" width="30%" height="30%"></p>

  - <img src="https://latex.codecogs.com/svg.image?a_{out}" title="a_{out}" />에는 non-linear한 sigmoid 함수를 사용하여 linear한 MF모델보다 더 많은 표현이 가능하며, <img src="https://latex.codecogs.com/svg.image?h^{T}" title="h^{T}" />에는 latent vector의 영향력을 조절하는 non-uniform 값을 주어 내적할때 각 텀에 다른 가중치를 준다. 
 
  - 이때, <img src="https://latex.codecogs.com/svg.image?a_{out}" title="a_{out}" />에 identity function(항등함수)를 사용하고, <img src="https://latex.codecogs.com/svg.image?h^{T}" title="h^{T}" />를 uniform vector로 값을 주게 되면 MF 모델과 동일해진다. 

<br/>

### 3.3 Multi-Layer Perceptron (MLP)
- MLP는 사용자와 아이템 간의 concatenated vector를 여러 hidden layers를 통과시킨다.

- GMP는 linear하고 fixed element-wise product를 하기 때문에 사용자와 아이템 간의 복잡한 관계를 학습하기는 힘들지만, MLP는 GMP에 비해 더 많은 층을 사용하기 때문에 flexible하고 non-linearity한 특성으로 인해 더 복잡한 관계를 학습할 수 있다.

- 이에 따라 MLP는 다음과 같이 파라미터라이징한다. 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/155169489-7d8be19e-73c0-40f3-b9e8-371fd3cc369d.PNG" width="40%" height="40%"></p>

<br/>

### 3.4 Fusion of GMF and MLP
- 본 논문에서는 아래와 같이 GMF와 MLP를 합친 각자의 장점은 살리고 단점은 보완할 수 있는 모델을 제안하며, 이 모델을 Neural Matrix Factorization (NeuMF)로 명명하였다. 
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/155169493-d008460b-cd39-4ea8-9763-3485573f9aa6.PNG" width="50%" height="50%"></p>

- 선형적인 특성을 가진 GMF와 비선형적인 특성을 가진 MLP를 합쳐 사용자와 아이템 간의 더 복잡한 관계를 표현할 수 있다.

- 두 모델이 각각 다른 embedding을 학습하고 마지막 hidden layer에서 출력되는 값들을 concat하여 최종 점수를 계산한다. MLP 활성화 함수에는 ReLU를 사용하였다. 
  - 같은 embeddig을 share하면 fused model의 성능을 제한할 수 있으나, 이러한 방식을 택함으로 인해 더욱 flexible한 모델이 된다.

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/155169500-ffd4803c-c54c-41f3-8e8c-6dcc7325a70f.PNG" width="40%" height="40%"></p>

<br/>

#### 3.4.1 Pre-training
- GMF와 MLP를 각각 pre-trained된 모델로 학습을 진행하였고, 이 후 두 모델을 concat한 NeuMF를 통해 예측값을 출력하였다. 
  - GMF와 MLP가 random initialization으로 convergence 할때까지 training한다. 
  - 훈련한 모델의 파라미터를 NeuMF에서 활용한다.
  - 알파 값을 이용해 가중치를 부여하며, 본 논문에서는 0.5를 사용하였다.
  - GMP와 MLP를 각각 Adam 옵티마이저를 적용하여 학습시켰고, NeuMF에는 SGD 옵티마이저를 적용하였다. 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/155169502-5093d589-8281-4edf-8f0a-a9544aab567e.PNG" width="20%" height="20%"></p>

<br/>

## 4. Experiment
- NCF모델이 implicit CF모델보다 더 좋은 성능을 보이는가? (RQ1)  --> NeuMF가 모든 경우에서 SOTA를 달성함

- 제안된 최적화 프레임워크(Logloss with Negative sampling)의 효과? (RQ2) --> NeuMF>MLP>GMF 순으로 logloss가 줄어듬

- Layer가 더 깊어지면 사용자와 아이템 간의 상호작용을 학습하는데 도움이 되는가? (RQ3) --> layer를 늘릴수록 성능이 높아짐

<br/>

## 6. Conclusion and Future Work
- 

<br/>
