# LLM from Scratch

---

## 🎯 스터디 목표

1. 📑 논문의 이론과 구현 코드를 연결하여 개념을 자세히 설명하는 콘텐츠 작성
2. 🧩 구현한 코드를 정리하여 LLM 연구를 위한 작은 패키지 구축
3. 🌟 (선택) HuggingFace Code Contribution

---

## 📆 주차별 커리큘럼

<details>
<summary><b>Week 0: LLM에 대한 넓고 얕은 지식</b></summary>
<br>
LLM에 대한 기초적인 이해와 전반적인 개념 학습
</details>

<details>
<summary><b>Week 1: Architecture and Generation</b></summary>
<br>

### 🔑 키워드
Causal LM, Decoding Strategy, LLM Inference / Generation

### 📚 학습 내용
- 전형적인 오픈소스 LLM의 아키텍처를 살펴봅니다.
- 원본 트랜스포머 디코더와 현대 LLM의 구성 요소를 비교합니다.
- 토크나이저와 언어 모델의 기본 개념을 이해합니다.

### 📖 참고 자료
- [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [Transformers Auto Classes](https://huggingface.co/docs/transformers/en/model_doc/auto)
- [Text Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)

### 💻 실습 과제
- `transformers` 라이브러리의 `Auto` 클래스의 동작 원리를 이해합니다.
- `forward`와 `generate` 메서드의 관계와 차이를 이해합니다.
- Forward Pass에서 Activation Memory가 어떻게 변하는지 살펴봅니다.
- 기본적인 디코딩 전략을 살펴보고, 샘플링에 관여하는 파라미터를 이해합니다.
</details>

<details>
<summary><b>Week 2: Embedding and RAG</b></summary>
<br>

### 🔑 키워드
Embedding, Attention, RAG

### 📚 학습 내용
- 언어 모델의 임베딩 레이어와 임베딩 모델의 차이를 살펴봅니다.
- RAG와 관련된 기본적인 개념을 이해합니다.

### 📖 참고 자료
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/1908.10084)
- [Getting Started with Embeddings](https://huggingface.co/blog/getting-started-with-embeddings)
- [Kanana Nano 2.1b Embedding](https://huggingface.co/kakaocorp/kanana-nano-2.1b-embedding)
- [Transformers RAG Documentation](https://huggingface.co/docs/transformers/en/model_doc/rag)

### 💻 실습 과제
- `forward` 메서드의 실행 과정에서 Hidden State와 Attention Map을 추출하는 방법을 이해합니다.
- `sentence_transformers` 라이브러리를 사용하여 텍스트 임베딩을 수행하고, 간단한 유사도 기반 검색을 실습합니다.
</details>

<details>
<summary><b>Week 3: Fine Tuning LLMs</b></summary>
<br>

### 🔑 키워드
Fine-Tuning, Supervised Fine Tuning, Chat Template

### 📚 학습 내용
- LLM의 학습 및 추론 원리를 이해하고, 파인튜닝 과정을 살펴봅니다.
- Base와 Chat 모델의 차이를 이해합니다.
- HuggingFace에서 데이터셋을 불러오고 조작하는 방법을 알아봅니다.
- Parameter Efficient Fine Tuning(PEFT)의 개념을 이해합니다.

### 📖 참고 자료
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [Instruction Tuning with GPT-4](https://arxiv.org/abs/2304.03277)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/index)
- [NLP Processing with Datasets](https://huggingface.co/docs/datasets/nlp_process)
- [PEFT Documentation](https://huggingface.co/docs/peft/index)
- [LoRA Developer Guide](https://huggingface.co/docs/peft/developer_guides/lora)
- [Chat Templating](https://huggingface.co/docs/transformers/main/chat_templating)
- [Meta Llama 3 Prompt Formats](https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/)

### 💻 실습 과제
- [Base Model](https://huggingface.co/meta-llama/Llama-3.2-1B)과 [Instruct Model](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)의 차이를 살펴봅니다.
- LLM에서 사용되는 Special Token의 역할을 이해합니다.
- `chat_template`을 사용하여 LLM 학습 데이터를 전처리합니다.
- `peft` 라이브러리를 사용하여 파라미터 효율적인 파인 튜닝을 수행합니다.
- `trainer` 클래스의 동작 원리를 이해합니다.
</details>

<details>
<summary><b>Week 4: LLM Quantization</b></summary>
<br>

### 🔑 키워드
Quantization, Post Training Quantization, Quantization Aware Training

### 📚 학습 내용
- LLM Compression의 여러 기법을 살펴봅니다.
- LLM 양자화 과정을 살펴봅니다.
- PTQ와 QAT의 차이를 비교합니다.
- Calibration의 개념을 이해합니다.

### 📖 참고 자료
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [Quantization in Hugging Face](https://huggingface.co/blog/merve/quantization)
- [BitsAndBytesConfig Documentation](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig)

### 💻 실습 과제
- `BitsAndBytesConfig`를 사용한 LLM 양자화 과정을 이해합니다.
- Calibration 과정을 살펴봅니다.
</details>

<details>
<summary><b>Week 5: Reinforcement Learning</b></summary>
<br>

### 🔑 키워드
Reinforcement Learning, RLHF, Preference Optimization

### 📚 학습 내용
- LLM을 활용한 강화학습이 어떻게 수행되는지 살펴봅니다.
- PPO, RLHF의 개념을 이해합니다.
- Preference Optimization 개념을 이해합니다.
- 메모리 관점에서 LLM RL이 어려운 이유와 극복 방안에 대해 살펴봅니다.

### 📖 참고 자료
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [TRL Documentation](https://huggingface.co/docs/trl/index)
- [PPO Trainer](https://huggingface.co/docs/trl/main/en/ppo_trainer)
- [DPO Trainer](https://huggingface.co/docs/trl/main/dpo_trainer)

### 💻 실습 과제
이번 주차 실습은 높은 메모리의 GPU를 필요로 하므로, 선택적으로 실습을 수행합니다.
- [LLM RL](https://github.com/huggingface/trl/blob/main/trl/models/modeling_value_head.py)의 구현 방식을 이해합니다.
- `PPOTrainer`, `DPOTrainer`를 사용한 LLM 학습을 수행합니다.
- PPO, DPO 각각에 사용하는 Reward Dataset, Preference Dataset을 분석합니다.
</details>

<details>
<summary><b>Week 6: Test Time Scaling</b></summary>
<br>

### 🔑 키워드
Test Time Compute Scaling, Large Reasoning Model

### 📚 학습 내용
- Chain of Thought 방법론을 이해합니다.
- LLM Reasoning과 Test Time Compute Scaling에 공부합니다.
- Reward Model을 사용한 생성 과정을 이해합니다.

### 📖 참고 자료
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [System 2 Attention (S2A)](https://arxiv.org/abs/2303.11569)
- [Search and Learn](https://huggingface.co/learn/cookbook/en/search_and_learn)

### 💻 실습 과제
- 다양한 프롬프팅 방법론을 사용해서 텍스트를 생성하고, 결과를 비교합니다.
- Best-of-N, Beam Search 등의 Test Time Compute Scaling을 직접 수행합니다.
</details>

<details>
<summary><b>Week 7: Long Context</b></summary>
<br>

### 🔑 키워드
Long Context LLM, RoPE, Needle in a Haystack

### 📚 학습 내용
- Attention의 동작 원리를 복습하고, LLM의 Context Length 개념을 이해합니다.
- LLM의 Context Length를 늘리기 위한 방법론을 살펴봅니다.
- Long Context LLM의 문제점과 극복 방안을 공부합니다.

### 📖 참고 자료
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Awesome LLM Long Context Modeling](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling)

### 💻 실습 과제
- Long Context와 관련된 연구를 조사한 후, 어떻게 구현되었는지 살펴봅니다.
- Long Context LLM에서 발생할 수 있는 문제(Needle in a Haystack)를 이해합니다.
</details>

<details>
<summary><b>Week 8: Efficient Inference</b></summary>
<br>

### 🔑 키워드
KV Caching, Prompt Compression, Speculative Decoding

### 📚 학습 내용
- 추론 단계에서 Transformer의 효율성을 개선하기 위한 방법을 살펴봅니다.
- KV Caching 개념을 이해합니다.
- Test Time Compute Scaling 과정에서 발생하는 문제를 해결하기 위한 기법을 조사합니다.

### 📖 참고 자료
- [Medusa: Simple LLM Inference Acceleration Framework](https://arxiv.org/abs/2311.10732)
- [SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference](https://arxiv.org/abs/2305.09781)
- [DeepSpeed: Accelerating Large-scale Model Inference](https://arxiv.org/abs/2211.05102)
- [KV Cache Documentation](https://huggingface.co/docs/transformers/en/kv_cache)
- [Understanding KV Caching](https://huggingface.co/blog/not-lain/kv-caching)
- [Speculative Generation](https://huggingface.co/docs/text-generation-inference/conceptual/speculation)
- [Generation Strategies](https://huggingface.co/docs/transformers/en/generation_strategies)

### 💻 실습 과제
- KV Caching 사용 여부에 따른 Inference 과정의 차이를 이해합니다.
- Efficient Inference 기법의 구현 방법을 살펴봅니다.
</details>
