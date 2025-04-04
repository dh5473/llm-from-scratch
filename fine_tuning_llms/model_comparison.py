import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd


def compare_base_instruct_models():
    """
    메타 LLaMA 베이스 모델과 인스트럭트 모델의 차이점을 비교합니다.
    같은 질문에 대한 두 모델의 응답을 비교하여 인스트럭트 모델이
    사용자 지시에 더 적합하게 응답하는지 확인합니다.
    """
    base_model_name = "meta-llama/Llama-3.2-1B"
    instruct_model_name = "meta-llama/Llama-3.2-1B-Instruct"

    print(
        f"⏳ 베이스 모델({base_model_name})과 인스트럭트 모델({instruct_model_name})을 비교합니다..."
    )

    # 베이스 모델과 토크나이저 로드
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )

    # 인스트럭트 모델과 토크나이저 로드
    instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_name)
    instruct_model = AutoModelForCausalLM.from_pretrained(
        instruct_model_name, torch_dtype=torch.float16, device_map="auto"
    )

    # 모델 구성 비교
    print("\n📊 모델 구성 비교:")
    base_config = base_model.config
    instruct_config = instruct_model.config

    # 두 모델의 주요 차이점 출력
    print(f"  - 베이스 모델 어휘 크기: {base_config.vocab_size}")
    print(f"  - 인스트럭트 모델 어휘 크기: {instruct_config.vocab_size}")

    # 기타 주요 파라미터 비교
    for param in ["hidden_size", "num_attention_heads", "num_hidden_layers"]:
        if hasattr(base_config, param) and hasattr(instruct_config, param):
            print(
                f"  - {param}: 베이스={getattr(base_config, param)}, 인스트럭트={getattr(instruct_config, param)}"
            )

    # 테스트 프롬프트로 두 모델의 출력 비교
    test_prompts = [
        "인공지능이란 무엇인가요?",
        "한국 역사에 대해 간략히 설명해주세요.",
        "파이썬으로 'Hello, World!'를 출력하는 코드를 작성해주세요.",
    ]

    print("\n🔄 두 모델의 응답 비교:")

    results = []

    for prompt in test_prompts:
        # 베이스 모델 응답 생성
        base_inputs = base_tokenizer(prompt, return_tensors="pt").to(base_model.device)
        base_output = base_model.generate(
            base_inputs["input_ids"], max_length=100, do_sample=True, temperature=0.7
        )
        base_response = base_tokenizer.decode(base_output[0], skip_special_tokens=True)

        # 인스트럭트 모델 응답 생성
        instruct_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        instruct_inputs = instruct_tokenizer(instruct_prompt, return_tensors="pt").to(
            instruct_model.device
        )
        instruct_output = instruct_model.generate(
            instruct_inputs["input_ids"],
            max_length=100,
            do_sample=True,
            temperature=0.7,
            attention_mask=instruct_inputs["attention_mask"],
        )
        instruct_response = instruct_tokenizer.decode(
            instruct_output[0], skip_special_tokens=True
        )

        # 결과 저장
        results.append(
            {
                "프롬프트": prompt,
                "베이스 모델 응답": base_response,
                "인스트럭트 모델 응답": instruct_response,
            }
        )

        print(f"\n📝 프롬프트: {prompt}")
        print(f"🤖 베이스 모델: {base_response[:150]}...")
        print(f"🧠 인스트럭트 모델: {instruct_response[:150]}...")

    # 비교 결과를 CSV로 저장
    df = pd.DataFrame(results)
    output_path = "model_comparison_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ 비교 결과가 {output_path}에 저장되었습니다.")


if __name__ == "__main__":
    compare_base_instruct_models()
