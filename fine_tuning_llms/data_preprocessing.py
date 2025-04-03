from transformers import AutoTokenizer
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def preprocess_with_chat_template():
    """
    LLM 파인튜닝을 위한 대화 데이터 전처리 방법을 시연합니다.
    chat_template을 사용하여 대화 턴을 구조화된 형식으로 변환합니다.
    """
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"⏳ {model_name}의 chat_template을 사용하여 대화 데이터를 전처리합니다...")

    # 샘플 데이터셋 생성 (실제로는 외부 파일에서 로드할 수 있음)
    sample_conversations = [
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

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 토크나이저의 chat_template 확인
    print("\n📝 토크나이저의 chat_template:")
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        print(tokenizer.chat_template)
    else:
        print(
            "  - 이 토크나이저에는 기본 chat_template이 없습니다. 사용자 정의 템플릿을 사용합니다."
        )
        # 사용자 정의 템플릿 정의
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}\n{% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] }}\n{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}\n{% endif %}\n{% endfor %}"
        print(f"  - 사용자 정의 템플릿:\n{tokenizer.chat_template}")

    print("\n🔄 대화 데이터를 전처리하고 토큰화합니다...")

    # 대화를 chat_template 형식으로 변환 및 토큰화
    processed_samples = []
    token_lengths = []

    for i, conversation in enumerate(sample_conversations):
        # 1. chat_template 적용
        formatted_chat = tokenizer.apply_chat_template(
            conversation["messages"], tokenize=False, add_generation_prompt=True
        )

        # 2. 토큰화
        tokenized_chat = tokenizer.encode(formatted_chat)
        token_lengths.append(len(tokenized_chat))

        # 예시 출력
        print(f"\n📊 대화 {i+1} 처리 결과:")
        print(f"  - 원본 메시지 수: {len(conversation['messages'])}")
        print(f"  - 포맷팅된 텍스트 길이: {len(formatted_chat)} 문자")
        print(f"  - 토큰 수: {len(tokenized_chat)}")

        # 처음 몇 줄만 출력
        formatted_lines = formatted_chat.split("\n")
        preview_lines = (
            "\n".join(formatted_lines[:6]) + "\n..."
            if len(formatted_lines) > 6
            else formatted_chat
        )
        print(f"\n🔍 포맷팅된 텍스트 미리보기:\n{preview_lines}")

        # 일부 토큰 ID와 해당 토큰 출력
        print("\n🔤 토큰화 미리보기 (처음 10개):")
        tokens = tokenizer.convert_ids_to_tokens(tokenized_chat[:10])
        for token_id, token in zip(tokenized_chat[:10], tokens):
            print(f"  - ID {token_id}: '{token}'")

        # 결과 저장
        processed_samples.append(
            {
                "대화 ID": i + 1,
                "원본 대화": conversation,
                "포맷팅된 텍스트": formatted_chat,
                "토큰화된 ID": tokenized_chat,
                "토큰 수": len(tokenized_chat),
            }
        )

    # 토큰 길이 통계 계산
    avg_tokens = np.mean(token_lengths)
    max_tokens = np.max(token_lengths)
    min_tokens = np.min(token_lengths)

    print(f"\n📊 토큰 길이 통계:")
    print(f"  - 평균 토큰 수: {avg_tokens:.1f}")
    print(f"  - 최대 토큰 수: {max_tokens}")
    print(f"  - 최소 토큰 수: {min_tokens}")

    # 데이터 저장
    with open("processed_conversations.json", "w", encoding="utf-8") as f:
        json.dump(processed_samples, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 전처리된 데이터가 processed_conversations.json에 저장되었습니다.")

    # 샘플 훈련 데이터셋 형식 설명
    print("\n📚 파인튜닝을 위한 데이터셋 구조화 방법:")
    print("  1. 토큰화된 ID를 입력 및 레이블로 분리 (마스킹)")
    print("  2. attention_mask 생성 (패딩 토큰 위치 표시)")
    print("  3. 배치 처리를 위한 패딩 추가")
    print("  4. DataLoader를 통한 미니배치 구성")

    # 파인튜닝을 위한 데이터 구성 예시
    print("\n🧩 파인튜닝 데이터셋 예시 형식:")
    print(
        """  {
    "input_ids": [1, 2, 3, 4, 5, ...],       # 토큰화된 입력 ID
    "attention_mask": [1, 1, 1, 1, 1, ...],  # 패딩이 아닌 위치는 1, 패딩은 0
    "labels": [1, 2, 3, 4, 5, ...]           # 다음 토큰 예측을 위한 레이블
  }"""
    )


if __name__ == "__main__":
    preprocess_with_chat_template()
