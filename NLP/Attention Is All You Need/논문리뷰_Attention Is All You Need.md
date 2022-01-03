# Attention Is All You Need

본 논문은 2018년에 구글 리서치팀이 NIPS(Neural Information Processing Systems)에서 발표한 논문으로, 저자들은 Recurrence와 Convolutions를 제거하고, 오로지 Attention에 기반하여 설계된 Transformer라는 simple한 sequence transduction model 구조를 제안한다. 자연어처리(NLP)의 발전에 아주 큰 영향을 끼친 Transformer에 관한 논문으로, 최신 고성능 모델들은 Transformer Architecture를 기반으로 한다. 예를 들면, GPT는 Transformer의 Decoder 부분을 활용하고 BERT는 Transformer의 Encoder 부분을 활용한다. 

<br/>

## Abstract 
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

<br/>

## 1. Introduction
- RNN, LSTM, 그리고 GRU가 언어모델링과 기계번역에서 SOTA로 자리 잡아왔다. 하지만, Recurrent model의 이전 결과를 입력으로 받는 sequential한 특성은 두 가지 문제가 있다. 첫 번째, 연속적으로 이어지는 것이기 때문에 학습에서 병렬처리를 배제한다. 두 번째, sequence가 길어질 수록 메모리에 문제가 생긴다. 

- Attention 메커니즘은 input과 output의 길이의 관계 없이 dependency한 모델링을 할 수 있어서 sequence modeling과 transduction modeling에 중요한 요소를 차지해왔다. 하지만, 대부분의 경우에 Attention은 RNN과 함께 사용된다. 

- 본 논문에서는, Recurrence를 제외하고 input과 output의 global dependecy를 찾는 attention 매커니즘에 전적으로 기반하는 Transformer를 제안한다. Transformer 구조는 병렬처리를 가능하게 하고 번역의 질 등의 성능에 있어서 새로운 SOTA를 달성할 수 있다. 

<br/>

## 2. Background
- Sequential한 연산을 줄이기 위해 다양한 연구들(Extended Neural GPU, ByteNet, ConvS2S)이 제안되었다.
  - CNN을 활용하여 input과 output의 위치에 대한 hidden representation을 병렬로 계산하는 방식을 통해 효율성을 증대시켰다. 
  - 하지만, 이런 모델들에서는 input과 output을 연결하기 위해 필요로 하는 연산량은 거리에 따라서 증가한다. 따라서, distant position에 있는 dependency를 학습하기가 어렵다. (거리가 멀어질 수록 학습이 어려움)

- Transformer에서는 attention-weighted position을 평균을 함으로 인해 효율성은 떨어질수 있지만, number of operation을 상수로 고정시켜서 연산량을 감소시킨다.
  - 줄어든 효율성은 Multi-Head Attention 방식으로 상쇄할 수 있다.

- Self-attention은 representation of sequence를 계산하고자 single sequence에 있는 다른 position들을 연결시키는 attention 매커니즘이다. 
  - 지문이해, 요약 등의 다양한 task들에서 성공적으로 사용되고 있다. 

- RNN 또는 CNN 없이 self-attention 만으로 input과 output의 representation을 구한 모델은 Transformer가 처음이다. 

<br/>

## 3. Model Architecture
- sequence data를 다루는 많은 모델들은 encoder-decoder 구조를 가진다. 

  - Encoder는 input sequence (x<sub>1</sub> , ..., x<sub>n</sub>)를 continuous representation인 z = (z<sub>1</sub>, ... , z<sub>n</sub>)으로 변환한다. 
  
  - z가 주어지면, decoder는 output sequence (y<sub>1</sub>, ... , y<sub>n</sub>)를 하나씩 생성한다.  

- Transformer도 마찬가지로 이러한 encoder-decoder 구조를 가지는데, self-attention과 point-wise fully connected layer를 쌓아 만든 encoder와 decoder로 구성되어있다. 

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147875664-ab331085-dce4-4f20-83d4-4178a1843953.PNG" width="50%" height="50%"></p>


### 3.1 Encoder and Decoder Stacks
### Encoder
- Encoder는 N=6개의 동일한 layer로 구성되어 있다. 

- 각 layer는 multi-head self-attention mechanism과 position-wise fully connected feed-foward로 구성된 2개의 sub-layer를 가지고 있다. 

- 각 sub-layer는 residual connection으로 연결하였고, 이후 normalization을 한다. 

  - 따라서, sub-layer를 통과할 때마다 결과값으로 LayerNorm(x + sublayer(x))를 출력한다. 
  
- residual connection을 수월하게 하기 위해, 모델의 모든 sub-layer 및 임베딩 layer는 512개의 차원으로 output을 생성한다. 

<br/>

### Decoder
- Decoder 역시 N=6개의 동일한 layer로 구성되어 있으며, 각 sub-layer를 residual connection으로 연결하고 이후 normalization을 한다. 

- 2개의 sub-layer 외에도, decoder는 Encoder 스택의 출력을 통해 multi-head attetion을 수행하는 세번째 sub-layer를 가진다. 

- 또한, decoder의 self-attention sub-layer에서는 현재 위치보다 뒤에 있는 요소에 Attention 하는 것을 막기 위해 masking을 추가한다. 

  - 이는, i번째 position의 예측이 i의 이전의 output에만 의존하도록 만들어준다. (즉, 앞에 있는 단어로만 예측하고 뒤에 있는 단어를 미리 알지 못하도록)

<br/>

### 3.2 Attention
- Attention은 한 문장 내에서 특정 단어를 이해하려고 할때 어떤 단어들을 중점적으로 봐야 단어를 더 잘 이해할 수 있을지에 관한 것이다. 

- Attention fuction은 query와 key-value 쌍을 output에 맵핑한다.

  - Query(Q, 영향을 받는 단어), Key(K, 영향을 주는 단어), Values(V, 영향에 대한 가중치)는 모두 vector 형태이다.
  
- Output은 value의 가중치 합으로 계산 되는데, 각각의 value에 맞는 가중치는 query와 그에 맞는 key의 compatibility function에 의해 계산된다.

  - I Love You 라는 단어가 있을때, I라는 단어가 I, Love, You 각각에 대해 얼마큼의 연관성을 가지는지를 알아보고자 한다면, Query는 I, Key는 I, Love, You, Value는 예를 들면 0.2, 0.3, 0.5가 된다. 

<br/>

### Scaled Dot-Product Attention
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147882453-384d5f18-0b88-4985-a355-474547a3bead.png" width="30%" height="30%"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147881003-3e255556-c51e-4a7a-aac8-cec4bbc8a0ca.PNG" width="50%" height="50%"></p>

- 입력으로는 d<sub>k</sub> 차원의 query, key와 d<sub>v</sub> 차원의 value를 가진다. query와 key들의 dot product를 구하고, ![image](https://user-images.githubusercontent.com/79245484/147881693-9fd8cedf-1529-43f6-bca4-1476850262c0.png) (scaling factor) 로 나눈 값에 softmax 를 적용하여 value에 대한 weight를 얻는다. 마지막으로, value와 weight를 곱해주어 최종적인 Attention Value를 얻는다. 

- 대표적인 방법으로는 additive attention과 dot-product attention이 있다.

  - additive attention과 dot-product attention은 이론적으로 유사한 복잡성을 가지지만, 후자가 더 빠르고 공간적으로 효율적이다. 
  
  - 본 논문에서 사용하는 Scaled dot-product attention은 dot-product attention에 scaling factor를 추가한 것만 제외하면 둘은 동일하다. 
    - d<sub>k</sub>의 값이 클 때는 softmax가 매우 작은 gradient 값을 가지는 것을 방지하기 위해 scaling factor를 추가함으로 성능 향상에 도움이 된다.

<br/>

### Multi-Head Attention
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147882456-defcf992-9cd0-44dc-9a20-741dcc476ac2.png" width="30%" height="30%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147883351-c8fda6bc-6129-4ea0-a2a3-7af31dfb0c67.PNG" width="50%" height="50%"></p>

- d<sub>model</sub>-dimensional keys, values, queries를 가지고 single attention을 수행하는 것보다, key, value, query에 대해서 서로 다르게 학습된 d<sub>k</sub>, d<sub>v</sub>, d<sub>q</sub> 차원의 linear project를 h번 수행하는 것이 더 효과적이다. (벡터들의 크기를 줄이고 병렬처리가 가능하기 때문)

- 이렇게 project된 key, value, query에 대해 attention을 병렬적으로 계산해 d<sub>v</sub> 차원의 output 값을 얻게 된다. 이 값들은 concatenate되고 project 연산을 거쳐 최종 값을 산출한다. 

- Multi-Head Attention을 사용해서 서로 다른 위치에 있는 representation subspace 들로부터 정보를 얻을 수 있다. 
  - 각 head별로 차원 감소로 인해, 전체 계산 비용은 완전 차원을 가진 single-head Ateention과 비슷하다.  

<br/>

### Applications of Attention in our Model
Transformer에서는 3가지 방식으로 multi-head attention을 사용한다.

- encoder-decoder attention layer (Decoder 파트)

  - query는 이전 decoder에서 가져오고, key와 value는 encoder의 output에서 가져온다. 
  
  - 따라서, decoder의 모든 position에서 input sequence의 전체 position에 대해 attention 수행이 가능하다.
  
- Self-attention layer (Encoder 파트)

  - key, value, query가 모두 encoder의 이전 layer에서 나온 output이다. 
  
  - encoder의 각각의 position에서 이전 encoder layer의 모든 position에 대해서 attention 수행이 가능하다. 
  
- Self-attention layer(Decoder 파트)

  - Encoder의 Self-attention layer와 동일하지만, auto-regressive 속성을 보존하기 위해 output을 생성할 때 leftward information flow를 차단한다. 
  
  - 즉, 해당 position 이전까지의 모든 position만 attention이 가능하도록 하며, 미래 시점의 단어들 또는 output에는 접근하지 못하도록 한다. --> 현재 토큰 이후 값들에 대해서 masking out  

<br/>


### 3.3 Position-wise Feed-Forward Networks
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147884893-2d631fdc-0357-48c3-9dbc-467bb013799a.PNG" width="30%" height="30%"></p>


- Attention sub-layers 외에도, encoder와 decoder의 각 layer는 fully connected feed-foward network를 포함하는데, 이것은 각 position마다 개별적으로 그리고 동일하게 적용된다. 

- Position-wise Feed-Forward Network에서는 2번의 linear transformation을 수행하고, 두 linear transformation 사이에는 RELU 연산을 수행한다. 

<br/>


### 3.4 Embeddings and Softmax
- 대부분의 sequence transduction modele들과 같이 input/output tokens들을 벡터로 변환하는데 learned embedding을 사용하였다.

- 또한, Deder output이 예측된 다음 토큰의 확률을 계산하기 위해 learned linear transformation과 softmax function을 사용한다. 

- 두 개의 embedding layer와 pre-softmax linear transformation은 공통된 가중치 매트릭스를 가진다. 

<br/>


### 3.5 Positional Encoding
- Transformer는 recurrence나 convolution을 포함하지 않기 때문에, sequence의 순서 정보를 전달할 수 없다.

- 따라서, sequence에 있는 토큰의 상대적/절대적인 position에 대한 정보를 주입하기 위해 Positional encoding을 진행하였다. (단어별로 position에 대한 정보를 추가) 

- 이를 위해서, encoder 및 decoder 스택하단에 있는 input embedding에 positional encoding을 추가한다. positional encoding은 input embedding과 같은 d<sub>model</sub> 차원이므로, 두 값은 더해질 수 있다. 
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147884895-10595ba7-15fb-40f6-b416-681dd73e6a43.PNG" width="50%" height="50%"></p>

- Transformer에서는 sine과 cosine처럼 주기를 가지는 함수를 사용하며, 아주 작은 값을 position 마다 두고 그 값을 embedding마다 더해준다. 위 식을 이용해서 positonal encoding을 하고 위치 정보를 주입할 수 있다. (pos는 position, i는 차원)

- 성능의 차이는 크지 않았으나, 학습에서 접하지 못한 sequence가 들어왔을 때도 적절한 처리를 할 수 있는 Sinusoidal 방식을 하였다. 



<br/>


## 4. Why Self-Attention
Recurrent / Convolution 과 비교해서 Self-attention을 사용한데는 세 가지 이유가 있다.

- 첫 번째는, 레이어당 전체 계산 복잡도기 줄어든다. 

- 두 번째는, 병렬화가 가능한 연산이 늘어난다.
  - number of sequential operation 이 필요한 최소값으로 확인 가능하다. 

- 세 번째는, Long range dependency가 용이하다. 
  - Attention을 통해 모든 부분을 확인하니, RNN보다 훨씬 먼 거리에 있는 sequence를 더욱 잘 학습할 수 있다. 즉, 단어와 단어 사이가 길때(길이가 긴 문장) 더 학습이 용이하다. 

<br/>

이 세 가지를 정리한 표는 다음과 같다. 

<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147884894-ec562441-b6ce-4978-977f-cb8e2bedb831.PNG" width="80%" height="80%"></p>

<br/>

- 위 세가지 외에도, Attention을 사용하면 모델 자체의 동작을 해석하기 쉽다는 이점도 있다. 

<br/>

## 5. Training
### 5.1 Training Data and Batching
- 4.5 million 문장 pair로 구성되어 있는 WMT 2014 English-German dataset과, 36 million 문장 pair로 구성되어 있는 WMT 2014 English-French datasaet을 사용했다. 

### 5.2 Hardware and Schedule
- 8개의 NVIDIA P100 GPU가 있는 기계에서 모델을 훈련시켰다. 

- base 모델은 각 training step 마다 0.4초가 걸렸고(10,000steps), 12시간동안 학습했다.

- big 모델은 각 training step 마다 1.0초가 걸렸고(300,000steps), 3.5일동안 학습했다. 

### 5.3 Optimizer
- Adam optimizer를 사용하였고, <img src="https://latex.codecogs.com/svg.image?\beta&space;_{1}&space;=&space;0.9,&space;\beta&space;_{2}&space;=&space;0.98&space;&space;\&space;and&space;\&space;&space;\epsilon&space;=&space;10^{-9}&space;" title="\beta _{1} = 0.9, \beta _{2} = 0.98 \ and \ \epsilon = 10^{-9} " /> 를 사용했다. 

- learning rate는 아래의 공식에 따라 변화하며, warmup_step까지는 linear하게 learning rate를 증가시켰다가, warmup_step 이후에는 step_num의 inverse square root에 비례하게 감소시킨다. 본 논문에서는 warmup_steps = 4000으로 설정하였다. 
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147884896-a5f8ed5d-a046-4a34-97fa-cfc2c48f112a.PNG" width="80%" height="80%"></p>

<br/>

### 5.4 Regularization
#### Residual Dropout
- 각 sub-layer의 output이 sub-layer의 input으로 사용되거나 normalized가 되기 전에 dropout을 적용했다.

- 추가적으로, encoder와 decoder 스택 사이에 embedding과 positional encoding을 더하여 droput을 적용했다. 

- <img src="https://latex.codecogs.com/svg.image?P&space;_{drop}&space;=&space;0.1&space;" title="P _{drop} = 0.1 " /> 를 사용했다. 

#### Label Smoothing
- 학습이 진행되는 동안, <img src="https://latex.codecogs.com/svg.image?\epsilon_{ls}=0.1&space;" title="\epsilon_{ls}=0.1 " /> 의 label smoothing 값을 적용했다. 

  - 보통 딥러닝에서 softmax를 학습할 경우에는 레이블을 원-핫 인코딩으로 전환해준다.
  
  - 하지만, 이 방식은 정답과 오답을 이분화하여 나타내는 것이 아니라 정답은 1에 가까운 값 / 오답은 0에 가까운 값, 즉 0~1 사이 값으로 표현하여 모델이 너무 학습데이터에 치중하여 학습하지 못하도록 보완하는 방법이다. 
  
  - 이는, 모델의 perplextity를 해치기는 하지만, accuracy와 BLEU score를 개선시켰다. 


<br/>


## 6. Results
### 6.1 Machine Translation
- WMT 2014 English-German 번역에서 big transformer model이 앙상블을 포함한 이전 SOTA 모델보다 2.0 BLEU로 앞서며, new SOTA(BLEU score of 28.4)를 달성했다. 

  - base model 역시 training 비용을 고려했을 때 이전 모델들을 뛰어넘었다.  

- WMT 2014 English-French 번역에서도 big transformer model이 이전의 다른 single model보다 학습시간은 1/4로 줄었음에도 불구하고 BLUE score(41.0)는 더 뛰어났다. 
  - 학습 효율과 성능 둘 다 개선되었음  

<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147884897-c31c6a31-76ca-44e8-a775-564595db4ab9.PNG" width="80%" height="80%"></p>


<br/>

### 6.2 Model Variations
- Transformer의 components의 중요도를 평가하기 위해 English-German 모델을 newstest2013이라는 새로운 데이터에 적용해보면서, base model을 다양하게 변형시켰다. 

  - 연산량은 유지하면서 attention의 head와 key,value의 차원을 조절해보았다 --> head가 너무 적은 것도, 많은 것도 성능을 악화시켰다. (head=8 일 때 성능이 가장 좋음)
  
  - Attention key size를 줄이는 것도 성능에 악영향을 주었다. 
  
  - 모델의 size를 키우면 성능이 향상되었다.
  
  - dropout이 오버피팅 방지에 효과적이다 (=성능향상) 
  
  - 위치에 대한 정보를 주기 위해 sinusoids(Sine, Cosine 함수를 이용한 encoding) 대신 positional embedding을 사용했을 때는 base model과 비슷한 성능을 보인다. 

<br/>

<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147884898-8bc88a33-66eb-49a2-8d12-1c99bfa2718b.PNG" width="80%" height="80%"></p>


<br/>

## 7. Conclusion
- 본 연구에서, encoder-decoder 구조에서 가장 일반적으로 사용되는 recurrent layer를 multi-head attention으로 대체하면서, attention만 사용한 최초의 sequence 변환 모델인 transformer를 제시했다. 

- 번역 과제의 경우 Transformer는 recurrent 또는 convolutional layer 기반 구조보다 훨씬 빠르게 학습하고 더 좋은 성능을 보여주었다. 

  - 계산량이 줄고(RNN은 순차적인 계산으로 속도가 느림) 병렬화를 적용하여(Multi-head로 병렬로 계산가능) 학습 속도가 매우 빠르다.

- Attention에 기반한 모델을 텍스트 뿐만 아니라, 오디오/이미지/영상 등의 상대적으로 큰 입출력을 필요로 하는 task에도 적용을 할 예정이다. 

  - 즉, 특정 Task에 족송적이지 않고 general하게 사용이 가능할 것이다. 

- Generation을 덜 Sequential 하게 만드는 것이 또 다른 연구 목적이다.
