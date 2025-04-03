from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd


def explore_special_tokens():
    """
    LLM에서 사용되는 스페셜 토큰의 역할을 탐색합니다.
    스페셜 토큰은 모델에게 특정 문맥이나 지시사항을 전달하는 데 사용됩니다.
    예) <|user|>, <|assistant|>, <|system|>, <s>, </s>, <|endoftext|> 등
    """
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    print(f"⏳ {model_name}의 스페셜 토큰을 분석합니다...")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 기본 스페셜 토큰 분석
    print("\n📋 기본 스페셜 토큰:")

    special_tokens = {
        "BOS (Beginning of String)": tokenizer.bos_token,
        "EOS (End of String)": tokenizer.eos_token,
        "UNK (Unknown)": tokenizer.unk_token,
        "PAD (Padding)": tokenizer.pad_token,
        "SEP (Separator)": (
            tokenizer.sep_token if hasattr(tokenizer, "sep_token") else "없음"
        ),
        "CLS (Classification)": (
            tokenizer.cls_token if hasattr(tokenizer, "cls_token") else "없음"
        ),
        "MASK": tokenizer.mask_token if hasattr(tokenizer, "mask_token") else "없음",
    }

    # 스페셜 토큰 ID 출력
    tokens_info = []
    for name, token in special_tokens.items():
        token_id = tokenizer.convert_tokens_to_ids(token) if token != "없음" else "없음"
        print(f"  - {name}: '{token}' (ID: {token_id})")
        tokens_info.append({"토큰 유형": name, "토큰": token, "토큰 ID": token_id})

    # 토큰 추가 테스트
    print("\n🧪 새로운 스페셜 토큰 추가 테스트:")

    # 기존 어휘 크기 확인
    original_vocab_size = len(tokenizer)
    print(f"  - 원래 어휘 크기: {original_vocab_size}")

    # 새로운 스페셜 토큰 추가
    new_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|thinking|>"]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    # 어휘 크기 변화 확인
    print(f"  - 추가된 토큰 수: {num_added}")
    print(f"  - 새로운 어휘 크기: {len(tokenizer)}")

    # 새로 추가된 토큰 ID 확인
    print("\n🔍 새로 추가된 스페셜 토큰 ID:")
    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  - '{token}' → ID: {token_id}")
        tokens_info.append(
            {"토큰 유형": "추가된 토큰", "토큰": token, "토큰 ID": token_id}
        )

    # 예시 문장 토큰화 테스트
    print("\n🔤 스페셜 토큰을 포함한 텍스트 토큰화 예시:")

    test_texts = [
        "일반 텍스트입니다.",
        f"{tokenizer.bos_token}시작 토큰이 있는 텍스트입니다.{tokenizer.eos_token}",
        f"<|system|>\n지시에 따라 대답하세요.\n<|user|>\n한국의 수도는 어디인가요?\n<|assistant|>\n서울입니다.",
    ]

    for i, text in enumerate(test_texts):
        # 토큰화
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)

        print(f"\n예시 {i+1}: {text}")
        print(f"  - 토큰 수: {len(tokens)}")
        print(
            f"  - 토큰: {tokens[:10]}..." if len(tokens) > 10 else f"  - 토큰: {tokens}"
        )
        print(
            f"  - 토큰 ID: {token_ids[:10]}..."
            if len(token_ids) > 10
            else f"  - 토큰 ID: {token_ids}"
        )

    # 스페셜 토큰의 임베딩 영향 설명
    print("\n📚 스페셜 토큰의 중요성:")
    print("  1. 모델 입출력 경계 표시: BOS/EOS는 텍스트의 시작과 끝을 표시")
    print("  2. 대화 구조화: <|user|>, <|assistant|>는 대화 참여자 구분")
    print("  3. 파인튜닝 성능 향상: 명확한 역할 구분으로 모델 응답 품질 개선")
    print(
        "  4. 토크나이저 변경 시 주의: 스페셜 토큰을 추가하면 모델 임베딩도 조정 필요"
    )

    # 결과 저장
    df = pd.DataFrame(tokens_info)
    output_path = "special_tokens_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ 스페셜 토큰 분석 결과가 {output_path}에 저장되었습니다.")


if __name__ == "__main__":
    explore_special_tokens()
