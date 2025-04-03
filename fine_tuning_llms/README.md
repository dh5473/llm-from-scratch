# 🚀 LLaMA 모델 파인튜닝 실습

이 프로젝트는 Meta의 LLaMA 모델을 이용한 파인튜닝 실습 코드를 제공합니다. 베이스 모델과 인스트럭트 모델의 차이점 분석, 스페셜 토큰 활용, chat_template을 이용한 데이터 전처리, PEFT 라이브러리를 활용한 효율적인 파인튜닝 등을 다룹니다.

## 📋 주요 학습 내용

1. **베이스 모델과 인스트럭트 모델 비교**
   - LLaMA 3.2 1B 베이스 모델과 인스트럭트 모델 간의 차이 분석
   - 동일 질문에 대한 두 모델의 응답 비교

2. **스페셜 토큰의 역할 이해**
   - 언어 모델에서 사용되는 `<|user|>`, `<|assistant|>` 등 스페셜 토큰의 의미와 활용
   - 토크나이저에 새로운 스페셜 토큰 추가 방법

3. **chat_template을 활용한 데이터 전처리**
   - 대화 데이터를 모델 학습에 적합한 형태로 변환
   - Hugging Face의 `apply_chat_template` 활용

4. **PEFT를 이용한 파라미터 효율적 파인튜닝**
   - LoRA(Low-Rank Adaptation) 방식으로 모델 효율적 학습
   - 전체 파라미터 대신 일부만 학습하는 기법 적용

5. **Trainer 클래스의 동작 원리 이해**
   - Hugging Face Trainer API의 내부 동작 메커니즘
   - 학습 과정 시각화 및 커스터마이징 방법

## 🛠️ 설치 방법

필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

## 📊 프로젝트 구조

```
fine_tuning_llms/
├── model_comparison.py        # 베이스와 인스트럭트 모델 비교
├── special_tokens.py          # 스페셜 토큰 탐색
├── data_preprocessing.py      # chat_template 데이터 전처리
├── fine_tuning.py             # PEFT 라이브러리로 모델 파인튜닝
├── trainer_explained.py       # Trainer 클래스 동작 원리 설명
├── run_fine_tuning.py         # 전체 과정 실행 스크립트
├── requirements.txt           # 필요 패키지 목록
└── README.md                  # 프로젝트 설명
```

## 🚀 실행 방법

### 전체 과정 한번에 실행

```bash
python run_fine_tuning.py
```

### 개별 태스크 실행

```bash
# 베이스와 인스트럭트 모델 비교
python run_fine_tuning.py --task compare

# 스페셜 토큰 탐색
python run_fine_tuning.py --task special_tokens

# 데이터 전처리 방법 탐색
python run_fine_tuning.py --task preprocess

# Trainer 클래스 동작 이해
python run_fine_tuning.py --task explain

# 파인튜닝 실행
python run_fine_tuning.py --task train --model meta-llama/Llama-3.2-1B-Instruct --epochs 3 --batch_size 1 --lr 2e-5
```

## 📝 주요 파라미터

파인튜닝 실행 시 다음 파라미터를 활용할 수 있습니다:

- `--model`: 사용할 기본 모델 (기본값: meta-llama/Llama-3.2-1B-Instruct)
- `--dataset`: HuggingFace 데이터셋 (기본값: 샘플 데이터셋)
- `--dataset_path`: 커스텀 데이터셋 경로
- `--output_dir`: 결과 저장 경로 (기본값: ./results)
- `--epochs`: 학습 에포크 수 (기본값: 1)
- `--batch_size`: 배치 크기 (기본값: 1)
- `--lr`: 학습률 (기본값: 2e-5)
- `--lora_r`: LoRA 랭크 (기본값: 8)
- `--load_8bit`: 8비트 양자화 사용 여부

## 📋 파인튜닝 고급 활용법

### 1. 커스텀 데이터셋 사용

JSON 형식의 자체 데이터셋 사용:

```bash
python run_fine_tuning.py --task train --dataset_path path/to/your/dataset.json
```

### 2. 모델 양자화

GPU 메모리가 제한된 환경에서 8비트 양자화 활용:

```bash
python run_fine_tuning.py --task train --load_8bit
```

### 3. LoRA 하이퍼파라미터 조정

LoRA 랭크 및 학습률 조정:

```bash
python run_fine_tuning.py --task train --lora_r 16 --lr 5e-5
```

## 📈 결과 분석 방법

파인튜닝 실행 후 다음 파일들을 확인할 수 있습니다:

- `model_comparison_results.csv`: 베이스/인스트럭트 모델 응답 비교
- `special_tokens_analysis.csv`: 스페셜 토큰 분석 결과
- `processed_conversations.json`: 전처리된 대화 데이터
- `results/final_model/`: 파인튜닝된 최종 모델

## 📚 학습 리소스

- [Hugging Face Transformers 문서](https://huggingface.co/docs/transformers/index)
- [PEFT 라이브러리 문서](https://huggingface.co/docs/peft/index)
- [LLaMA 3.2 모델 정보](https://huggingface.co/meta-llama/Llama-3.2-1B)

## ⚠️ 주의사항

- 파인튜닝에는 충분한 GPU 메모리가 필요합니다. 
- GPU 메모리가 제한적인 경우 `--load_8bit` 옵션을 사용하여 양자화를 활성화하세요.
- 결과는 사용하는 모델 버전과 데이터셋에 따라 달라질 수 있습니다. 