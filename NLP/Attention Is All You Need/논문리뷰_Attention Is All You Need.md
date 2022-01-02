# Attention Is All You Need

본 논문은 2018년에 구글 리서치팀이 NIPS(Neural Information Processing Systems)에서 발표한 논문으로, 자연어처리(NLP)의 발전에 아주 큰 영향을 끼친 Transformer에 관한 논문이다. 

## Abstract 
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.


## 1. Introduction

## 2. Background

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
