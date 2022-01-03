# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## BERT
본 논문은 2019년 구글에서 발표한 논문으로, 현재 사용하는 모든 언어 모델의 기반이 되는 BERT에 대한 논문이다. Attention is all you need에서 나온 Transformer 구조를 활용한 모델로써, NLP 분야에 큰 발전을 가져왔다. 

<br/>

## 1. Abstract
We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation
models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications. 

BERT is conceptually simple and empirically powerful. It  btains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

<br/>

## 2. Introduction
- NLP에서 pre-training은 효과적이라는 것을 보여주었으며, pre-trained language representation을 down-streak tasks에 적용하는데는 2가지 방법이 있다.
  - Feature-based
    - Task-specific architectures를 사용함으로, 기존의 input에 pre-trained representation을 feature로서 추가한다. 
    - Ex) ELMo (단방향의 LSTM으로 구성된 두 개의 모델을 붙여 양방향으로 학습하도록 하는 모델, Shallow concatenation, left-to-right & right-to-left)

  - Fine-tunning
    -  최소한의 task-specific parameters만을 추가하고, 모든 pre-trained parameter들을 fine-tunning 함으로써 downstream task에서 학습한다. 
    -  Feature-based와 달리 fine-tunning 과정에서 pre-trained feature도 갱신된다. 
    -  Ex) GPT (token이 이전의 것에만 참여할 수 있음)


- 하지만, ELMo와 GPT 같은 경우에는 pre-trained에서 단방향의 architecture만 사용하여 학습을 진행한다는 limitation이 있다. 
  - OpenAI GPT의 경우에는 left to right 구조를 가지고 있는데, 이는 각각의 token이 이전에 나타난 token들에만 attend 할 수 있다. 따라서, 문장 단위 Task에서 최적의 성능을 얻지 못하고, question answering 같은 token에 민감한(문맥을 양쪽에서 모두 이해해야하는) Task에는 좋지 않은 성능을 보일 수 있다.  
  - 즉, 한 방향으로만 문맥을 파악하는 것은 충분하지 않고 성능을 제한한다. 


- 본 논문에서는, 양방향성을 고려한 fine-tunning based approach인 BERT(Bidirectional Encoder Representations from Transformers)를 제안한다. 
  - BERT는 Masked Language Model(MLM)을 사용함으로 일부 tokens from the input을 마스크하고, original 단어를 예측하는 방식으로 학습을 진행한다. 이러한 학습 과정에서 MLM은 왼쪽과 오른쪽에 존재하는 모든 토큰을 고려하게 되는데, 이러한 점이 양방향 학습을 적용하는 것이다. 
  - 또한, BERT는 Next Sentence Prediction을 사용하는데, 이는 문장 간의 연관성을 pre-training 시키는 것이다. 
  - 대용량의 unlabeled data로 pre-training 하고 특정 task에 대해 transfer learning 하는 모델이다. 


- 본 논문의 contribution은 다음과 같다.
  - Language representation에서 양방향 pre-training의 중요성을 증명했다. 
  - pre-trained representation이 heavily-engineered task-specific architectures의 필요성을 감소시켰고, Fine-tunning based representation 모델로서 처음으로 SOTA를 달성했다. 
  - BERT는 11개 NLP task에서 SOTA를 달성했다. 

<br/>

## 3. BERT
### Model Architecture
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147931233-ddaf27a2-b6ed-4060-9d0e-febd8556a326.PNG" width="80%" height="80%"></p>

- BERT는 크게 두 가지 단계로 나눠진다. 
  - Pre-training: 대규모 데이터셋을 바탕으로 문장 이해를 학습하는 과정으로, 모델은 특정 pre-training task에 상관없이 unlabeled data로부터 학습된다.  
  
  - Fine-tunning: pre-training 과정에서 생성된 parameter를 바탕으로 labeled data를 통해서 fine-tuning을 진행하며, 각각의 downstream task에 대해서 개별적인 fine-tuned 모델들이 생성된다. 

  - Pre-trained architecture와 final downstream architecture는 크게 다르지 않다 (Unified model). 마지막 encoder의 결과로 나온 token들을 어떻게 다루는지만 변한다. 

- BERT의 모델 구조는 Transformer의 encoder 부분을 활용하였으며, 2가지 다른 모델로 실험을 진행하였다.
  - Bert<sub>base</sub> : Layer = 12, Hidden size = 768, Multi-head Self Attention = 12, Total Parameter = 110M)
    - 비교를 위해, OpenAI GPT와 비슷한 모델을 가진다. (BERT=bidirectional self-attention, GPT=현재 토큰의 왼쪽에 있는 문맥만 참조 가능)
  - Bert<sub>Large</sub> : Layer = 24, Hidden size = 1024, Multi-head Self Attention = 16, Total Parameter = 340M)


<br/>

### Input/Output Representations
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147931235-2d2ffe79-9af2-495a-94f4-6d1401b4081a.PNG" width="80%" height="80%"></p>

- BERT를 여러 자연어 문제에 적용하기 위해서는 input으로 들어오는 sequence가 single sentence인지, pair of sentence (e.g. <Question, Answer>)를 나타내는지 명확하게 알 수 있어야 한다. 

- BERT의 input은 3가지 embedding의 합(Token Embdding + Segment Embeddding + Position Embedding)으로 이루어지며, WordPiece embedding을 사용한다.

- 모든 sequence의 첫 번째 token은 Classification token[CLS]로 시작된다. 
  - 전체의 transformer 층을 다 거치고 난 뒤의, [CLS]에 해당하는 마지막 hideen state는 classification tasks를 풀기 위한 것으로 모든 sequence 정보를 담고 있다.

- Sentence pair는 single sequence로 묶어져 입력되며, 문장을 구분하기 위해 2가지 방법을 사용한다.
  - 먼저, Special Token[SEP]을 사용해서 구분한다.
  - 그 후, 두 sentence 중 어디에 속한 token 인지를 나타내는 학습된 segment embedding을 각각의 token에 더해준다 --> 해당 token이 sentence A 또는 sentence B 어디에 속하는지 구분 가능해짐

<br/>

### 3.2 Pre-training BERT
### Masked LM
- 직관적으로, deep bidirectional model이 left-to-right model이나 left-to-right과 right-to-left 모델의 shallow concatenation 보다 성능이 좋을 수 밖에 없다. 

- deep bidirectional representation을 학습하기 위해, 다음과 같은 과정을 거친다.
  - WordPiece token(input)의 15%를 랜덤하게 masking 하고, masked word만 예측한다. 
   
  - Fine-tuning에는 [MASK] token이 없으므로, pre-training과 fine-tuning 사이에 mismatch가 발생한다. 이 문제점을 완화하기 위해 'masked' words를 [MASK] token으로만 대체 하지 않고, 다양한 방식으로 대체한다. 
  
    - 전체 WordPiece token 중에서 15% 중 80%는 [MASK] token으로 대체하고, 10%는 random token으로 대체하고, 10%는 바꾸지 않고 원래 token으로 유지한다. 
    - 이를 통해, Model이 모든 Token에 대해서 실제로 맞는 Token인지 의심을 하기 때문에 학습을 더 잘해낼수 있다.  

<br/>

### Next Sentence Prediction(NSP)
- 전체적 문맥을 이해하는 기법이다. 
- 50%는 실제 다음 문장 A, 50%는 corpus의 랜덤 문장 

<br/>

### Pre-training data
- 

<br/>

### 3.3 Fine-tunning BERT
- Self-attention 매커니즘을 이용해서 두 문장을 하나의 sequence로 인코딩한다.
- pre-training과 비교해서 fine-tuning은 비용이 더 적다. 











<br/>

## 4. Experiments
### 4.1 GLUE
- 

<br/>

### 4.2 SQuAD v1.1
- 

<br/>

### 4.3 SQuAD v2.0
-

<br/>

### 4.4 SWAG
- 

<br/>


## 5. Ablation Studies
### 5.1 Effect of Pre-training Tasks
- NPS를 뺏을 때의 성능이 떨어짐으로, NPS의 중요성을 확인할 수 있다. 

<br/>

### 5.2 Effect of Model Size
-


<br/>

### 5.3 Featured-based Approach with BERT
- 

<br/>

## 6. Conclusion
- 

<br/>
