#!/usr/bin/env python3
import argparse
import subprocess
import os


def print_header(message):
    """헤더 형식으로 메시지 출력"""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)


def run_command(command, description):
    """명령어 실행 및 결과 출력"""
    print_header(description)
    print(f"실행 명령어: {command}")
    print("-" * 80)
    result = subprocess.run(command, shell=True, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="LLM 파인튜닝 실습 실행 스크립트")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["compare", "special_tokens", "preprocess", "train", "explain", "all"],
        help="실행할 작업 선택 (기본값: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="파인튜닝에 사용할 모델 (기본값: meta-llama/Llama-3.2-1B-Instruct)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace 데이터셋 이름 (기본값: 샘플 데이터셋 사용)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="결과 저장 디렉토리 (기본값: ./results)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="학습 에포크 수 (기본값: 1)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="배치 크기 (기본값: 1)"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="학습률 (기본값: 2e-5)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA 랭크 (기본값: 8)")
    parser.add_argument("--load_8bit", action="store_true", help="8비트 양자화 사용")
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="로컬 데이터셋 경로"
    )

    args = parser.parse_args()

    print_header("🌟 LLaMA 모델 파인튜닝 실습 시작 🌟")

    # 작업 목록 결정
    tasks = []
    if args.task == "all":
        tasks = ["compare", "special_tokens", "preprocess", "explain", "train"]
    else:
        tasks = [args.task]

    # 필요한 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    for task in tasks:
        if task == "compare":
            # 모델 비교 실행
            run_command(
                "python model_comparison.py", "베이스 모델과 인스트럭트 모델 비교"
            )

        elif task == "special_tokens":
            # 스페셜 토큰 탐색 실행
            run_command("python special_tokens.py", "LLM 스페셜 토큰 탐색")

        elif task == "preprocess":
            # 데이터 전처리 실행
            run_command(
                "python data_preprocessing.py",
                "chat_template을 사용한 대화 데이터 전처리",
            )

        elif task == "explain":
            # Trainer 설명 실행
            run_command("python trainer_explained.py", "Trainer 클래스 동작 원리 설명")

        elif task == "train":
            # 파인튜닝 실행
            command = f"python fine_tuning.py "
            command += f"--model_name {args.model} "
            command += f"--output_dir {args.output_dir} "
            command += f"--num_train_epochs {args.epochs} "
            command += f"--per_device_train_batch_size {args.batch_size} "
            command += f"--learning_rate {args.lr} "
            command += f"--lora_r {args.lora_r} "

            if args.load_8bit:
                command += "--load_in_8bit "

            if args.dataset:
                command += f"--dataset_name {args.dataset} "

            if args.dataset_path:
                command += f"--dataset_path {args.dataset_path} "

            run_command(command, "PEFT를 사용한 LLaMA 모델 파인튜닝")

    print_header("🎉 LLM 파인튜닝 실습이 모두 완료되었습니다!")
    print("\n실행된 작업:")
    for i, task in enumerate(tasks):
        print(f"  {i+1}. {task}")

    print("\n결과물 확인 방법:")
    print(f"  - 모델 비교 결과: model_comparison_results.csv")
    print(f"  - 스페셜 토큰 분석: special_tokens_analysis.csv")
    print(f"  - 전처리된 대화 데이터: processed_conversations.json")
    print(f"  - 파인튜닝된 모델: {args.output_dir}/final_model/")


if __name__ == "__main__":
    main()
