from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
import torch
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import os
import time


def explain_trainer_class():
    """
    Trainer 클래스의 동작 원리를 설명하고 시각화합니다.
    """
    print("📚 Hugging Face Trainer 클래스 동작 원리 설명 📚")
    print("=" * 70)

    # Trainer 클래스의 주요 컴포넌트 설명
    print("\n1️⃣ Trainer 클래스의 주요 컴포넌트")
    print("-" * 50)

    components = {
        "model": "학습할 모델",
        "args": "TrainingArguments 객체로 학습 설정 지정",
        "data_collator": "배치 생성 시 각 샘플을 처리하는 함수",
        "train_dataset": "학습 데이터셋",
        "eval_dataset": "평가 데이터셋 (선택 사항)",
        "tokenizer": "토큰화를 위한 토크나이저 (선택 사항)",
        "compute_metrics": "평가 메트릭 계산 함수 (선택 사항)",
        "optimizers": "최적화기와 스케줄러 (선택 사항)",
    }

    for component, description in components.items():
        print(f"  • {component}: {description}")

    # 학습 과정 설명
    print("\n2️⃣ Trainer 학습 워크플로우")
    print("-" * 50)

    workflow_steps = [
        "모델, 데이터셋, 학습 인자 초기화",
        "최적화기 및 학습률 스케줄러 설정",
        "데이터 로더 생성 및 배치 샘플링 준비",
        "학습 루프 시작",
        "배치 데이터 로드 및 전처리",
        "모델 순전파 (forward pass) 및 손실 계산",
        "역전파 (backward pass) 및 그래디언트 계산",
        "그래디언트 누적 (설정된 경우)",
        "옵티마이저 스텝 및 학습률 업데이트",
        "로깅 및 체크포인트 저장 (설정된 간격에 따라)",
        "평가 루프 실행 (설정된 경우)",
        "학습 완료 후 최종 모델 저장",
    ]

    for i, step in enumerate(workflow_steps):
        print(f"  {i+1}. {step}")

    # 간단한 Trainer 사용 예제 시연
    print("\n3️⃣ 간단한 Trainer 사용 예제")
    print("-" * 50)

    # 데이터 샘플 생성
    print("  • 샘플 데이터셋 생성 중...")

    texts = [
        "인공지능은 인간의 학습능력과 추론능력, 지각능력을 인공적으로 구현한 컴퓨터 시스템입니다.",
        "머신러닝은 데이터로부터 패턴을 학습하는 알고리즘의 한 분야입니다.",
        "딥러닝은 여러 층의 인공신경망을 사용하여 데이터의 특징을 자동으로 학습합니다.",
        "자연어 처리는 인간의 언어를 컴퓨터가 이해하고 처리하는 기술입니다.",
        "LLM(Large Language Model)은 대규모 언어 모델로, 텍스트 생성 및 이해 능력이 뛰어납니다.",
    ]

    dataset = Dataset.from_dict({"text": texts})

    # 토크나이저 설정
    print("  • 토크나이저 초기화 중...")

    model_name = "meta-llama/Llama-3.2-1B"  # 실제 실행 시에는 작은 모델로 대체 가능

    try:
        # 실제 모델 다운로드 없이 설명만
        print(
            f"    (참고: 실제 모델({model_name})은 다운로드하지 않고 설명만 제공합니다)"
        )

        # 가상의 토크나이저와 모델 동작 설명
        print("  • 토크나이저가 텍스트를 토큰으로 변환하는 과정:")
        sample_text = "인공지능 모델을 학습합니다."
        print(f"    원본 텍스트: '{sample_text}'")
        print(
            f"    토큰화 결과 (예시): ['인공', '지능', '모델', '을', '학습', '합니다', '.']"
        )
        print(f"    토큰 ID (예시): [14235, 9876, 2468, 1357, 8642, 5791, 9]")

    except Exception as e:
        print(f"    모델을 로드할 수 없습니다: {e}")
        print("    설명을 위한 예시로 진행합니다.")

    # TrainingArguments 설명
    print("\n4️⃣ TrainingArguments 주요 파라미터")
    print("-" * 50)

    training_args_params = {
        "output_dir": "모델 체크포인트 저장 경로",
        "num_train_epochs": "학습 에포크 수",
        "per_device_train_batch_size": "디바이스당 학습 배치 크기",
        "per_device_eval_batch_size": "디바이스당 평가 배치 크기",
        "learning_rate": "초기 학습률",
        "weight_decay": "가중치 감쇠 비율",
        "logging_dir": "로그 저장 디렉토리",
        "logging_steps": "로깅 간격 (스텝 단위)",
        "save_steps": "모델 저장 간격 (스텝 단위)",
        "save_total_limit": "저장할 최대 체크포인트 수",
        "evaluation_strategy": "평가 전략 (no, steps, epoch)",
        "gradient_accumulation_steps": "그래디언트 누적 스텝 수",
        "fp16": "16비트 부동소수점 학습 활성화",
        "warmup_steps/warmup_ratio": "웜업 스텝 수 또는 비율",
    }

    for param, desc in training_args_params.items():
        print(f"  • {param}: {desc}")

    # 데이터 콜레이터 설명
    print("\n5️⃣ DataCollator의 역할")
    print("-" * 50)

    print("  • 데이터 콜레이터는 개별 샘플을 배치로 결합하는 함수입니다.")
    print("  • DataCollatorForLanguageModeling의 주요 기능:")
    print("    1. 다양한 길이의 시퀀스를 패딩하여 동일한 길이로 만듦")
    print("    2. 마스킹된 언어 모델링(MLM)에서 일부 토큰을 마스킹")
    print("    3. 인과적 언어 모델링(CLM)에서 입력을 레이블로 사용")
    print("    4. 배치 내 샘플들을 텐서로 변환")

    # 학습 루프 시각화
    print("\n6️⃣ 학습 루프 시각화")
    print("-" * 50)

    # 가상의 학습 데이터 생성
    epochs = 3
    steps_per_epoch = 5
    loss_values = []

    # 단순한 감소 패턴의 손실 값 생성
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            # 노이즈가 있는 감소 패턴
            loss = (
                2.5
                - (epoch * steps_per_epoch + step) / (epochs * steps_per_epoch) * 2.0
            )
            loss += np.random.normal(0, 0.1)  # 약간의 노이즈 추가
            loss_values.append(max(0.1, loss))  # 최소값 보장

    # 학습 루프 시각화 (ASCII 아트)
    print("  • 학습 루프 진행 과정 (예시):")
    print("  " + "-" * 40)
    print("  |  에포크  |  스텝  |  손실  |  학습률  |")
    print("  " + "-" * 40)

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            idx = epoch * steps_per_epoch + step
            loss = loss_values[idx]
            lr = 5e-5 * (1 - idx / (epochs * steps_per_epoch))  # 학습률 감소
            print(f"  |    {epoch+1}     |   {step+1}    | {loss:.4f} | {lr:.6f} |")

    print("  " + "-" * 40)

    # 손실 변화 시각화를 위한 텍스트 기반 그래프
    print("\n  • 손실 변화 (텍스트 시각화):")
    max_loss = max(loss_values)
    min_loss = min(loss_values)
    scale = 20  # 그래프 너비

    for epoch in range(epochs):
        print(f"  에포크 {epoch+1}: ", end="")
        for step in range(steps_per_epoch):
            idx = epoch * steps_per_epoch + step
            loss = loss_values[idx]
            # 손실 값을 0-scale 범위로 정규화하여 '#' 문자 개수 결정
            bars = (
                int((loss - min_loss) / (max_loss - min_loss) * scale)
                if max_loss > min_loss
                else 0
            )
            print("#" * bars + " " * (scale - bars) + f" {loss:.4f}", end="  ")
        print()

    # Trainer의 주요 메서드 설명
    print("\n7️⃣ Trainer 클래스의 주요 메서드")
    print("-" * 50)

    methods = {
        "train()": "학습 프로세스를 시작합니다.",
        "evaluate()": "평가 데이터셋에서 모델을 평가합니다.",
        "predict()": "테스트 데이터셋에 대한 예측을 수행합니다.",
        "save_model()": "모델을 저장합니다.",
        "log()": "지정된 로그 값을 기록합니다.",
        "create_optimizer_and_scheduler()": "옵티마이저와 스케줄러를 초기화합니다.",
        "compute_loss()": "모델의 손실을 계산합니다.",
        "training_step()": "단일 학습 스텝을 수행합니다.",
        "prediction_step()": "단일 예측 스텝을 수행합니다.",
    }

    for method, desc in methods.items():
        print(f"  • {method}: {desc}")

    # 학습 진행 상황 모니터링 방법
    print("\n8️⃣ 학습 진행 상황 모니터링")
    print("-" * 50)

    monitoring_tools = {
        "로깅": "logging_steps 인자를 통해 로깅 빈도 설정",
        "TensorBoard": "--report_to tensorboard 옵션으로 시각화 도구 활성화",
        "Weights & Biases": "--report_to wandb 옵션으로 W&B 통합",
        "커스텀 콜백": "TrainerCallback 클래스를 상속하여A 맞춤형 콜백 구현",
        "진행 상황 표시": "disable_tqdm=False 설정으로 진행 표시줄 활성화",
    }

    for tool, desc in monitoring_tools.items():
        print(f"  • {tool}: {desc}")

    # 맞춤형 Trainer 하위 클래스 예시
    print("\n9️⃣ 맞춤형 Trainer 하위 클래스 예시")
    print("-" * 50)

    custom_trainer_code = """class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 기본 구현 대신 커스텀 손실 계산 로직 추가
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 커스텀 손실 함수 (예: 레이블 스무딩 적용)
        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs):
        # 커스텀 학습 스텝 구현
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # 일반적인 손실 계산
        loss = self.compute_loss(model, inputs)
        
        # 추가 메트릭 계산 (예: 퍼플렉서티)
        perplexity = torch.exp(loss)
        
        # 로깅
        self.log({"loss": loss.detach().item(), "perplexity": perplexity.detach().item()})
        
        # 스케일링 및 역전파
        loss = self.accelerator.backward(loss)
        
        return loss.detach()"""

    print(f"```python\n{custom_trainer_code}\n```")

    # 결론
    print("\n🔟 Trainer 클래스 요약")
    print("-" * 50)

    print(
        "  • Trainer 클래스는 Hugging Face Transformers에서 모델 학습을 단순화합니다."
    )
    print("  • 주요 장점:")
    print("    - 복잡한 학습 루프를 추상화하여 코드 작성량 감소")
    print("    - 분산 학습, 혼합 정밀도 학습 등 다양한 기능 내장")
    print("    - 평가, 로깅, 체크포인트 저장 등 편의 기능 제공")
    print("    - 커스터마이징을 통해 특수한 요구사항 구현 가능")
    print("  • 학습 단계 요약:")
    print("    1. 모델, 데이터셋, 학습 인자 초기화")
    print("    2. 옵티마이저와 스케줄러 구성")
    print("    3. 배치 처리 및 학습 루프 실행")
    print("    4. 정기적인 로깅 및 체크포인트 저장")
    print("    5. 평가 및 최종 모델 저장")


if __name__ == "__main__":
    explain_trainer_class()
