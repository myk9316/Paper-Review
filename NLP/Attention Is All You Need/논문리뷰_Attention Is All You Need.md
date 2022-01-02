# Attention Is All You Need

본 논문은 2018년에 구글 리서치팀이 NIPS(Neural Information Processing Systems)에서 발표한 논문으로, 자연어처리(NLP)의 발전에 아주 큰 영향을 끼친 Transformer에 관한 논문이다. 
저자들은 Recurrence와 Convolutions를 제거하고, 오로지 Attention에 기반하여 설계된 Transformer라는 새롭고 simple한 구조를 제안한다. 

## Abstract 
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.


## 1. Introduction
- RNN, LSTM, 그리고 GRU가 언어모델링과 기계번역에서 SOTA로 자리 잡아왔다. 하지만, Recurrent model의 이전 결과를 입력으로 받는 sequential한 특성은 두 가지 문제가 있다. 첫 번째, 연속적으로 이어지는 것이기 때문에 학습에서 병렬처리를 배제한다. 두 번째, sequence가 길어질 수록 메모리에 문제가 생긴다. 

- Attention 메커니즘은 input과 output의 길이의 관계 없이 dependency한 모델링을 할 수 있어서 sequence modeling과 transduction modeling에 중요한 요소를 차지해왔다. 하지만, 대부분의 경우에 Attention은 RNN과 함께 사용된다. 

- 본 논문에서는, Recurrence를 제외하고 input과 output의 global dependecy를 찾는 attention 매커니즘에 전적으로 기반하는 Transformer를 제안한다. Transformer 구조는 병렬처리를 가능하게 하고 번역의 질 등의 성능에 있어서 새로운 SOTA를 달성할 수 있다. 




## 2. Background
- Sequential한 연산을 줄이기 위해 다양한 연구들(Extended Neural GPU, ByteNet, ConvS2S)이 제안되었다.
  - CNN을 활용하여 input과 output의 위치에 대한 hidden representation을 병렬로 계산하는 방식을 통해 효율성을 증대시켰다. 
  - 하지만, 이런 모델들에서는 input과 output을 연결하기 위해 필요로 하는 연산량은 거리에 따라서 증가한다. 따라서, distant position에 있는 dependency를 학습하기가 어렵다. (거리가 멀어질 수록 학습이 어려움)

- Transformer에서는 attention-weighted position을 평균을 함으로 인해 효율성은 떨어질수 있지만, number of operation을 상수로 고정시켜서 연산량을 감소시킨다.
  - 줄어든 효율성은 Multi-Head Attention 방식으로 상쇄할 수 있다.

- Self-attention은 representation of sequence를 계산하고자 single sequence에 있는 다른 position들을 연결시키는 attention 매커니즘이다. 
  - 지문이해, 요약 등의 다양한 task들에서 성공적으로 사용되고 있다. 

- RNN 또는 CNN 없이 self-attention 만으로 input과 output의 representation을 구한 모델은 Transformer가 처음이다. 



## 3. Model Architecture
- sequence data를 다루는 많은 모델들은 encoder-decoder 구조를 가진다. 
  - Encoder는 input sequence (x<sub>1</sub> , ..., x<sub>~n</sub>)를 continuous representation인 z = (z<sub>1</sub>, ... , z<sub>n</sub>)으로 변환한다. 
  - z가 주어지면, decoder는 output sequence (y<sub>1</sub>, ... , y<sub>n</sub>)를 하나씩 생성한다.  

- Transformer도 마찬가지로 이러한 encoder-decoder 구조를 가지는데, self-attention과 point-wise fully connected layer를 쌓아 만든 encoder와 decoder로 구성되어있다. 

<img src="https://user-images.githubusercontent.com/79245484/147875664-ab331085-dce4-4f20-83d4-4178a1843953.PNG" width="50%" height="50%"/>



### 3.1 Encoder and Decoder Stacks
#### Encoder
- Encoder는 N=6 개의 동일한 layer로 구성되어 있다. 
  - 각 layer는 multi-head self-attention mechanism과 position-wise fully connected feed-foward로 구성된 2개의 sub-layer를 가지고 있다. 
    - 각 sub-layer는 residual connection으로 연결하였고, 이후 normalization을 한다. 
- 임베딩 Layer와 모든 sub-layer의 output의 dimension은 512이다. 


#### Decoder
- 


### 3.2 Attention
### 3.3 Position-wise Feed-Forward Networks
### 3.4 Embeddings and Softmax
### 3.5 Positional Encoding

## 4. Why Self-Attention

## 5. Training
### 5.1 Training Data and Batching
### 5.2 Hardware and Schedule
### 5.3 Optimizer
### 5.4 Regularization

## 6. Results
### 6.1 Machine Translation
### 6.2 Model Variations

## 7. Conclusion

## 8. Advice / limitation
