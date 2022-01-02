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
  - 이러한 연구들은, CNN을 활용하여 input과 output의 위치에 대한 hidden representation을 병렬로 계산하는 방식을 통해 효율성을 증대시켰다. 
  - 하지만, 이런 모델들에서는 input과 output을 연결하기 위해 필요로 하는 연산량은 거리에 따라서 증가한다. 따라서, distant position에 있는 dependency를 학습하기가 어렵다. (거리가 멀어질 수록 학습이 어려움)

- Transformer에서는 attention-weighted position을 평균을 함으로 인해 효율성은 떨어질수 있지만, number of operation을 상수로 고정시켜서 연산량을 감소시킨다.
  - 줄어든 효율성은 Multi-Head Attention 방식으로 상쇄할 수 있다고 한다.

## 3. Model Architecture
### 3.1 Encoder and Decoder Stacks
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
