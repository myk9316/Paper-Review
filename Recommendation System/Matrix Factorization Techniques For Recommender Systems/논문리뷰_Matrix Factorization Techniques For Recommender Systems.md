# Matrix Factorization Techniques For Recommender Systems
- 
<br/>

## Introduction
- 시장에 너무나도 많고 다양한 제품이 제공됨에 따라, 소비자들은 선택의 폭이 넓어졌다. 따라서, 소비자에게 적절한 상품을 매칭하는 것이 소비자의 만족도와 충성도를 높이기 위한 중요한 일이 되었다. 이에 따라, 많은 업체들이 사용자들의 흥미와 관심사를 분석하여 개인화된 추천을 제공하는 추천시스템에 더욱 관심을 가지게 되었다. 

<br/>

## 1. Recommender System Stratgies
- 추천시스템은 크게 Content-based filtering(콘텐츠 기반 필터링)과 Collaborative Filtering(협업 필터링)으로 구분할 수 있다.

### Content-based Filtering
- 각 사용자 또는 아이템에 대한 프로필을 만들고, 그 특성을 구체화시킨다. 
  - 사용자 프로필 예시: 인구통계학적 정보, 특정 질문에 대한 답변
  - 아이템 프로필 예시: 영화 장르/참여배우/박스 오피스 인기
- 이 방법은 종종 수집할 수 없거나 수집하기 어려운 외부 정보를 수집해야한다.  
- 이 방법을 사용하여 성공적인 결과를 낸 예시로는 Music Genome Project가 있다. 

<br/>


### Collaborative Filtering
- 사용자와 아이템 간의 상호 상관 관계를 분석하여 새로운 사용자-아이템 관계를 찾아주는 것으로, Explicit한 프로필을 만들지 않고 구매 기록이나 구매 평점 같은 사용자의 과거 행동에만 의존한다. 

- Domain Free로 특정 domain에 대한 지식이 필요없는 장점이 있는 반면, 새로운 아이템과 사용자에 대해서 Cold Start Problem 이라는 문제가 존재한다. 

- Collaborative Filtering은 또 2가지 방식으로 나뉜다.
  - **Neighborhood method (근접이웃 방식)**
    - 사용자간 혹은 아이템 간의 유사성을 계산하는 것에 중점을 둔다. 
    - 동일한 아이템에게 비슷한 평가를 내린 아이템들은 서로 이웃이 되고, 동일한 사용자에게 비슷한 평가를 받은 아이템들은 서로 이웃이 된다. 
    - 아이템 중심 방법(item-based approach): 특정 아이템에 대한 사용자의 선호도를 비슷한 다른 아이템을 해당 사용자가 어떻게 평가했는지에 따라 평가된다. 
    - 사용자 중심 방법(user-based approach): 유사도가 높은 사용자들은 동일한 아이템에 대해서 비슷한 평가를 한다. 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/148513875-318e4fb3-a4fc-464e-9c0c-5b5194b95673.PNG" width="50%" height="50%"></p>

<br/>

  - **Latent Factor Model (잠재요인 방식)**
    - 사용자와 아이템 간의 유사성을 계산하는 것에 중점을 둔다.  
    - 사용자의 아이템에 대한 평가 패턴에서 추론된 20~100개의 요인들을 이용하여 아이템들과 사용자들을 특성화해서 점수를 설명하는 방식이다. 
      - 영화를 예시로 하면, 아이템의 factor들은 코미디vs드라마, 액션의 양, 시청연령 / 사용자의 factor들은 해당 영화의 요인들에서 높은 점수를 받은 영화를 얼마나 좋아하는지 측정 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/148513878-c88fd1b3-97ba-4b04-ba89-70e6d04e1683.PNG" width="50%" height="50%"></p>

<br/>

## 2. Matrix Factorization Methods
- Latent Factor Model을 가장 성공적으로 구현하는 방법 중 하나는 **Matrix Factorization(MF)** 이다. 

- MF는 아이템의 평점 패턴으로부터 추론한 요인 벡터들을 통해 아이템과 사용자를 특성화하고, 이때 아이템과 사용자 요인간의 강한 관련성이 있으면 추천이 이루어진다. 

- 이 방법은 좋은 확장성과 예측 정확도, 그리고 다양한 실생활에 적용할 수 있는 유연성으로 인해 많은 인기를 얻었다. 

- 추천시스템은 입력 데이터의 다양한 유형에 의지하는데, 그 중 가장 좋은 데이터는 좋아요/싫어요 혹은 별점의 수 같은 높은 품질의 **명시적(explicit) 피드백**이다. 일반적으로 이러한 피드백은 적은 비율로 이루어지기 때문에, 명시적 피드백은 Sparse Matrix 이루게 된다. 

- MF의 장점은 추가적인 정보의 통합을 허용한다는 것이다. 즉, 명시적 피드백을 사용할 수 없을 때, 추천시스템은 사용자의 구매이력, 브러우저 기록, 검색 패턴, 마우스 움직임 등의 **암시적(implicit) 피드백**을 사용하여 해당 사용자의 선호를 파악할 수 있다. 암시적 피드백은 주로 Dense Matrix로 표현된다. 

<br/>

## 3. A Basic Matrix Factorization Model
- MF 모델은 사용자와 아이템 모두를 차원 f의 공동 잠재 요인 공간에 매핑하는데, 사용자-아이템 간의 상호작용은 해당 공간에서 내적으로 모델링 된다. 
- 각각의 아이템 i는 q<sub>i</sub>로, 사용자 u는 p<sub>u</sub>라는 벡터 표현된다. q<sub>i</sub>와 p<sub>u</sub>의 내적의 결과는 사용자-아이템 사이의 상호작용을 반영하며, 이는 아이템에 대한 사용자의 전반적인 관심도를 표현한다. 이를 수식으로 나타내면 아래와 같다. 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/148522790-0ecd351f-ada8-4b7b-a8cd-9d3afb61e20b.PNG"></p>

- 이때, 주요 challenge는 각 아이템과 사용자를 요인 벡터에 매핑하는 것인데, 이 과정만 끝내면 사용자가 아이템을 어떻게 평가할 것인지 쉽게 알 수 있다. 

- 이러한 모델은 SVD(Singular Value Decomposition)과 매우 유사하지만, 추천시스템에서는 결측값의 존재로 인해 적용하기가 힘들다.
  - 적은 수의 알려진 항목만 사용하는 것은 과적합을 일으킬 수 있으나, 결측값을 채워넣는 것은 비효율적이고 데이터를 왜곡 시킬 수 있다.
  - 따라서, 최근 연구에서는 관찰된 평점만을 직접적으로 모델링하는 방법이 제시되었으며, 이때 규제화를 통해 과적합을 방지하였다. 

- 요인 벡터(p<sub>u</sub> 와 q<sub>i</sub>)를 알기 위해, 시스템은 아래의 식처럼 관측된 평점 세트를 통해 정규화된 squared error를 최소화한다. 여기서, k는 r<sub>ui<sub/>이 관측됐을 때의 (u, i)의 세트를 의미한다. 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/148557996-9836110b-b63d-4732-938d-c07595b8637d.PNG"></p>

- 결국 이 모델에서 하고자 하는 것은, 알려지지 않은 평점을 예측하는 것이기 때문에 규제화를 통해 과적합을 방지해야 한다. 이때 규제항이 필요한데, <img src="https://latex.codecogs.com/svg.image?\lambda&space;" title="\lambda " /> 가 규제의 정도를 제어하며, <img src="https://latex.codecogs.com/svg.image?\lambda&space;" title="\lambda " />는 주로 cross-validation을 통해 값이 결정된다. 


<br/>

## 4. Learning Algorithms
-

<br/>

### Stochastic gradient descent
-

<br/>

### Alternating least squares
-

<br/>

## 5. Adding Biases
-

<br/>

## 6. Additional Input Sources
-

<br/>

## 7. Temporal Dynamics
-

<br/>

## 8. Inputs With Varing Confidence Levels
-

<br/>

## 9. Netflix Prize Competition
- 

<br/>
