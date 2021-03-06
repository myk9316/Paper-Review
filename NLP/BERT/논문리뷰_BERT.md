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
  - 전체의 transformer 층을 다 거치고 난 뒤의, [CLS]에 해당하는 마지막 hideen state는 classification tasks를 풀기 위한 것으로 모든 sequence 정보를 담고 있다. (앞에서 모든 입력 전체를 바라봄)

- Sentence pair는 single sequence로 묶어져 입력되며, 문장을 구분하기 위해 2가지 방법을 사용한다.
  - 먼저, Special Token[SEP]을 사용해서 구분한다.
  - 그 후, 두 sentence 중 어디에 속한 token 인지를 나타내는 학습된 segment embedding을 각각의 token에 더해준다 --> 해당 token이 sentence A 또는 sentence B 어디에 속하는지 구분 가능해짐

<br/>

### 3.1 Pre-training BERT
### Masked LM
- 직관적으로, deep bidirectional model이 left-to-right model이나 left-to-right과 right-to-left 모델의 shallow concatenation 보다 성능이 좋을 수 밖에 없다. 

- deep bidirectional representation을 학습하기 위해, 다음과 같은 과정을 거친다.
  - WordPiece token(input token)의 15%를 랜덤하게 masking 하고, masked word만 예측한다. 그 후에, softmax를 통해 vocabulary에서 최종 token을 가져오는 방식으로 동작한다.  
   
  - Fine-tuning에는 [MASK] token이 없으므로, pre-training과 fine-tuning 사이에 mismatch가 발생한다. 이 문제점을 완화하기 위해 'masked' words를 [MASK] token으로만 대체 하지 않고, 다양한 방식으로 대체한다. 
  
    - 전체 WordPiece token 중에서 15% 중 80%는 [MASK] token으로 대체하고, 10%는 random token으로 대체하고, 10%는 바꾸지 않고 원래 token으로 유지한다. 
    - 이를 통해, 어떠한 단어를 예측해야될지 모르기 때문에 Model이 모든 Token에 대해서 실제로 맞는 Token인지 의심을 하게 되며, 학습을 더 잘해낼수 있다.  

<br/>

### Next Sentence Prediction(NSP)
- Question Answering과 Natural Language Inference 같은 task 들은 token 단위보다 sentence 간의 관계가 더 중요하다. 

- BERT는 sentence 간의 관계를 학습시키기 위해 NSP라고 불리는 Binary classification을 pre-train 시킨다. 

- 학습 과정에서 모델은 A, B 를 입력으로 받는다. 
  - 이 때, 50% 의 경우 B는 실제로 A의 다음 문장으로 구성되고, 50%는 corpus에서 임의의 sentence로 구성된다. 
  - B가 실제로 다음 문장일 경우는 IsNext, 임의의 문자일 경우에는 NotNext라고 labeling 한다. 

- 단순한 작업처럼 보이지만, 이러한 작업은 실제로 성능 향상에 아주 효과적이다. 

<br/>

### Pre-training data
- 본 연구에서 pre-training corpus로 다음과 같은 데이터를 사용했다. 
  - BooksCorpus (800M words)
  - English Wikipedia (2,500M words)

<br/>

### 3.2 Fine-tunning BERT
- Transformer의 self-attention 구조가 BERT를 많은 downstream task에 적용할 수 있도록 하기 때문에, BERT의 fine-tuning 과정은 매우 간단하다. 
(BERT 모델은 가져가고 해당 모델의 appropriate한 input과 output을 swapping out만 하면됨) 

- 보통 text-pair task에서는 문자열 쌍의 관계를 알아내기 위한 bidirectional cross attention을 적용하기 전에, 입력으로 들어온 sentence 각각에 대한 encoding을 먼저 수행해야 한다. 

- 반면, BERT의 경우에는 두 문장의 입력을 하나의 sequence로 생성해서 모델의 입력으로 제공한다. 그 후에 self-attention을 수행하기 때문에, 이 self-attention 과정 안에 두 문장 사이의 bidirectional cross attention이 이미 포함되어 있어서 별도의 처리가 필요 없다. 
  - 즉, 입력이 한 문장이든 여러 문장이든 관계없이 단일 모델, 같은 fine-tuning 방법으로 간단하게 처리할 수 있다.  

- 각 task에 대해 알맞는 input과 output을 BERT에 적용하고, 파라미터들은 해당 task에 맞게 end-to-end로 학습한다. 

- Token representation은 sequence tagging이나 question answering 같은 token level task에 사용하기 위해 output layer에 보낸다. 

- 문장 제일 앞에 위치한 [CLS] representation은 entailment 또는 sentiment analysis 같은 classification task에 사용된다. 

- Pre-training과 비교 했을 때, Fine-tuning 단계는 비교적 적은 비용으로 수행할 수 있다. 

<br/>

## 4. Experiments
본 연구는 BERT의 검증을 위해 11개의 NLP task에서 fine-tuning을 진행하였음

### 4.1 GLUE (The General Language Understanding Evaluation)
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147931237-e2249652-12b2-4a3a-a764-53fe3b11f3ca.PNG" width="80%" height="80%"></p>

- BERT가 모든 task에 대해 기존 모델들을 능가한다.
- BERT larger가 BERT base를 모든 task에서 뛰어나다. 

<br/>

### 4.2 SQuAD v1.1  (Question, Answer pair data set)
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147931239-e00b81f6-0cb1-448f-8f57-48980cd32765.PNG" width="40%" height="40%"></p>

- Wikipedia에서 추출하여 만든 질의응답 벤치마크로, 지문과 질문이 주어지면 모델은 지문안에서 맞는 답을 찾아야 한다. 
- BERT single model이 top ensemble model을 능가한다.
- BERT ensemble model이 top leaderboard model을 능가한다. 

<br/>

### 4.3 SQuAD v2.0  (Question, Answer pair data set)
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147931240-173d0c89-416b-4169-ba22-769e9d50156e.PNG" width="40%" height="40%"></p>

- SQuAD v2.0은 v1.1의 확장판으로 주어진 paragraphs에 정답이 없는 경우도 포함시켜서 좀 더 현실적인 학습을 할 수 있다.
- BERT는 이전의 best system보다 +5.1 F1 개선을 이루어냈다. 

<br/>

### 4.4 SWAG  (Situation With Adversarial Generations dataset(113K sentence-pair)
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147931241-f37637bc-382e-4f04-945f-90e098f72a88.PNG" width="40%" height="40%"></p>

- 4개의 선택지 중 주어진 sentence의 다음 상황으로 가장 알맞은 sentence 고르는 task 이다. 
- BERT가 SOTA를 달성한다. 

<br/>

## 5. Ablation Studies
### 5.1 Effect of Pre-training Tasks
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147931247-0499870a-3ff2-4850-9da6-3f98453d909a.PNG" width="40%" height="40%"></p>

- BERT vs No NSP : NSP 효과 관찰
  - MLM으로만 학습되고 NSP를 하지 않은 경우, MNLI, QUNI, SQUaD에서 성능이 떨어지는 것을 확인할 수 있다. 

- No NSP vs LTR & No NSP : Bidirectionality 효과 관찰
  - LTR 모델에서 모든 task에 대해 성능이 떨어지고, 특히 MRPC와 SQuAD task는 큰 폭으로 성능이 떨어졌다.
  - BiLSTM을 추가했을때, SQuAD의 경우에는 성능이 향상 되었다. (pre-trained birectional model에 비해서는 성능이 떨어짐) 

- ELMo 처럼 LTR과 RTL을 각각 학습해서 representation을 concantenate하는 방법도 있으나, 이 경우에는 bidirectional model에 비해 비용이 2배로 들고 less powerful 하다. 

<br/>

### 5.2 Effect of Model Size
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147931248-8fe42209-ae3f-4821-8faf-d42e99e6dc19.PNG" width="40%" height="40%"></p>

- larger model을 사용했을 때 모든 task에서 더 좋은 성능을 보여주었다. 

  - 특히, small dataset에서는 model size가 커지면 overfitting으로 인한 성능 감소가 일어날 수 있는데, BERT는 MRPC와 같은 작은 dataset에서도 larger model이 더 좋은 정확도를 보여주었다. 

- 즉, model size의 증가는 기계번역과 언어모델링 같은 large-scale task에서도 성능 향상에 기여를 하고, 충분한 pre-training이 있었다는 전제하에 small-scale task에서도 성능 향상에 기여한다. 

<br/>

### 5.3 Featured-based Approach with BERT
<p align="center"><img src="https://user-images.githubusercontent.com/79245484/147931251-3bd8de41-d41b-4520-a88f-b277a9feb7cf.PNG" width="40%" height="40%"></p>


- Feature-based : Fixed features are extracted from the pre-trained model

- BERT를 feature-based approach로도 사용할 수 있는데, 이러한 방식에는 몇 가지 장점이 있다.
  - Transformer encoder는 모든 task를 represent 하지 못하므로, task-specific model을 추가해서 사용해야 한다. 
  - Pre-compute를 통해서 training data의 representation을 생성하고, 적은 비용이 드는 모델을 해당 representation에서 여러번 학습함으로 Computational benefit을 얻을 수 있다. 
  
- BERT를 ELMo와 같이 마지막 layer에 Bi-LSTM을 부착시켜, 해당 layer만 학습시키는 방법론을 사용하여 실험을 했다.
  - 'Concat Last Four Hidden'만 사용하면, Fine-tunning approach와 0.3 F1 score 차이 밖에 나지 않는다.  --> BERT는 feature-based approach에도 효과적이다. 

<br/>

## 6. Conclusion
- BERT는 Transformer의 encoder 부분을 활용한 모델로, 이를 이용해 다양한 NLP task에 쉽게 적용이 가능하고 높은 성능을 보여주었다. 기존의 연구들과의 차이점은 bidirecitonal pre-training으로 좋은 품질의 pre-trained language representation을 얻음과 동시에 downstream-task에도 fine-tuning이 쉽게 가능한 점이다. 이를 통해, dataset의 사이즈의 관계없이, 다양한 NLP task에 대해 약간의 fine-tuning만으로 매우 우수한 성능을 보여주었다. 또한, language model에 있어, righ, unsupervised pre-training이 매우 중요한 요소라는 것을 입증하였다. 

- 저자들이 복잡한 수식없이 NLP 초심자들도 직관적으로 이해할 수 있도록 설득력 있게 주장을 펄쳐나가고, 여러 검증을 통해 매우 뛰어난 성능을 보여준 점이 굉장히 인상깊었다. 




<br/>
