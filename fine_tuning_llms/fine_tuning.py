import os
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset, Dataset
import argparse
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class FinetuningArguments:
    """
    파인튜닝에 필요한 인자들
    """

    model_name: str = field(
        default="meta-llama/Llama-3.2-1B-Instruct",
        metadata={"help": "사용할 기본 모델의 이름 또는 경로"},
    )
    dataset_name: str = field(
        default=None, metadata={"help": "사용할 HuggingFace 데이터셋 이름"}
    )
    dataset_path: str = field(
        default=None, metadata={"help": "자체 데이터셋의 로컬 경로"}
    )
    lora_r: int = field(default=8, metadata={"help": "LoRA의 랭크"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha 파라미터"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA 드롭아웃 비율"})
    output_dir: str = field(
        default="./results", metadata={"help": "결과물을 저장할 디렉토리"}
    )
    num_train_epochs: int = field(default=1, metadata={"help": "학습 에포크 수"})
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "디바이스당 학습 배치 크기"}
    )
    learning_rate: float = field(default=2e-5, metadata={"help": "학습률"})
    max_seq_length: int = field(default=512, metadata={"help": "최대 시퀀스 길이"})
    load_in_8bit: bool = field(
        default=False, metadata={"help": "모델을 8비트 정밀도로 로드"}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "모델을 4비트 정밀도로 로드"}
    )


def create_sample_dataset():
    """
    샘플 데이터셋 생성
    """
    # 간단한 샘플 데이터셋 준비
    conversations = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 도움이 되는 AI 어시스턴트입니다.",
                },
                {"role": "user", "content": "인공지능 모델 파인튜닝이란 무엇인가요?"},
                {
                    "role": "assistant",
                    "content": "인공지능 모델 파인튜닝은 사전 학습된 대규모 모델을 특정 작업이나 도메인에 맞게 조정하는 과정입니다. 전체 모델을 처음부터 학습시키는 것보다 효율적이며, 적은 양의 데이터로도 좋은 성능을 얻을 수 있습니다.",
                },
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 도움이 되는 AI 어시스턴트입니다.",
                },
                {
                    "role": "user",
                    "content": "파이썬에서 리스트와 튜플의 차이점은 무엇인가요?",
                },
                {
                    "role": "assistant",
                    "content": "파이썬에서 리스트와 튜플의 주요 차이점은 다음과 같습니다: 1) 가변성: 리스트는 가변(mutable)이고 튜플은 불변(immutable)입니다. 2) 문법: 리스트는 대괄호 [], 튜플은 소괄호 ()로 정의합니다. 3) 사용 용도: 리스트는 동일한 유형의 항목 컬렉션에 적합하고, 튜플은 관련된 다른 유형의 항목을 그룹화하는 데 적합합니다.",
                },
            ]
        },
    ]

    # 데이터셋 생성
    dataset = {"train": conversations}
    return dataset


def preprocess_data(examples, tokenizer, max_seq_length):
    """
    데이터 전처리 함수
    """
    # 토크나이저의 chat_template을 사용하여 메시지 포맷팅
    formatted_texts = []

    for conversation in examples:
        formatted = tokenizer.apply_chat_template(
            conversation["messages"], tokenize=False, add_generation_prompt=True
        )
        formatted_texts.append(formatted)

    # 데이터셋 생성
    dataset_dict = {"text": formatted_texts}
    dataset = Dataset.from_dict(dataset_dict)

    # 토큰화 함수 정의
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors=None,  # 개별 샘플에 텐서를 적용하지 않음
        )
        result["labels"] = result["input_ids"].copy()
        return result

    # 데이터셋에 토큰화 함수 적용
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    return tokenized_dataset


def fine_tune_model(args):
    """
    PEFT를 사용하여 LLaMA 모델 파인튜닝
    """
    print(f"⏳ {args.model_name} 모델 파인튜닝을 시작합니다...")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델 로드
    print("\n🔄 모델을 로드하는 중...")
    load_options = {}

    if args.load_in_8bit:
        load_options["load_in_8bit"] = True
        print("  - 8비트 양자화 사용")
    elif args.load_in_4bit:
        load_options["load_in_4bit"] = True
        print("  - 4비트 양자화 사용")
    else:
        load_options["torch_dtype"] = torch.float16
        print("  - float16 정밀도 사용")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", **load_options
    )

    # 양자화를 사용하는 경우 모델 준비
    if args.load_in_8bit or args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA 설정
    print("\n🔧 LoRA 설정 구성...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    print(f"  - LoRA 랭크: {args.lora_r}")
    print(f"  - LoRA 알파: {args.lora_alpha}")
    print(f"  - LoRA 드롭아웃: {args.lora_dropout}")

    # PEFT 모델 준비
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 데이터셋 로드
    print("\n📚 데이터셋 로드 및 전처리...")
    if args.dataset_name:
        dataset = load_dataset(args.dataset_name)
        print(f"  - HuggingFace 데이터셋 '{args.dataset_name}'을 로드했습니다.")
    elif args.dataset_path:
        dataset = load_dataset("json", data_files=args.dataset_path)
        print(f"  - 로컬 데이터셋 '{args.dataset_path}'을 로드했습니다.")
    else:
        print("  - 데이터셋 경로가 지정되지 않아 샘플 데이터셋을 생성합니다.")
        dataset = create_sample_dataset()

    # 데이터 전처리
    tokenized_dataset = preprocess_data(
        dataset["train"], tokenizer, args.max_seq_length
    )

    print(f"  - 데이터셋 크기: {len(dataset['train'])} 샘플")
    print(f"  - 최대 시퀀스 길이: {args.max_seq_length} 토큰")

    # 데이터 콜레이터 설정
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=100,
        save_total_limit=3,
        fp16=torch.cuda.is_available() and not (args.load_in_8bit or args.load_in_4bit),
        remove_unused_columns=False,
    )

    print("\n⚙️ 학습 설정:")
    print(f"  - 학습률: {args.learning_rate}")
    print(f"  - 에포크 수: {args.num_train_epochs}")
    print(f"  - 배치 크기: {args.per_device_train_batch_size}")
    print(f"  - 출력 디렉토리: {args.output_dir}")
    print(
        f"  - fp16 사용: {torch.cuda.is_available() and not (args.load_in_8bit or args.load_in_4bit)}"
    )

    # Trainer 초기화
    print("\n🚀 Trainer 초기화 및 학습 시작...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 학습 실행
    trainer.train()

    # 모델 저장
    final_output_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_output_dir, exist_ok=True)
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print(f"\n✅ 파인튜닝 완료! 모델이 {final_output_dir}에 저장되었습니다.")

    return model, tokenizer


def test_fine_tuned_model(model, tokenizer):
    """
    파인튜닝된 모델을 간단히 테스트
    """
    print("\n🧪 파인튜닝된 모델 테스트:")

    test_prompts = [
        "인공지능이란 무엇인가요?",
        "한국 역사에 대해 간략히 설명해주세요.",
        "파이썬으로 'Hello, World!'를 출력하는 코드를 작성해주세요.",
    ]

    for prompt in test_prompts:
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        print(f"\n📝 질문: {prompt}")
        print("🧠 모델 응답 생성 중...")

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 응답: {response}")


def main():
    """
    메인 함수
    """
    parser = HfArgumentParser(FinetuningArguments)
    args = parser.parse_args_into_dataclasses()[0]

    print("\n🌟 LLaMA 모델 파인튜닝 실습 🌟")
    print("=" * 50)

    # 디바이스 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️ 사용 디바이스: {device}")
    if device == "cuda":
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"  - 가용 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # 파인튜닝 실행
    model, tokenizer = fine_tune_model(args)

    # 모델 테스트
    test_fine_tuned_model(model, tokenizer)

    print("\n🎉 모든 과정이 완료되었습니다!")


if __name__ == "__main__":
    main()
