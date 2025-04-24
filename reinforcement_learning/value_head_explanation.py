"""
LLM에서 강화학습을 위한 Value Head 구현 방식 설명

이 파일은 Hugging Face의 TRL 라이브러리에서 구현된 Value Head의 작동 방식을 설명합니다.
참고: https://github.com/huggingface/trl/blob/main/trl/models/modeling_value_head.py
"""

import torch.nn as nn

"""
Value Head 구현의 핵심 아이디어

Value Head는 LLM에서 강화학습을 적용할 때 중요한 구성 요소로, 
주어진 상태(state)나 시퀀스의 가치(value)를 예측하는 네트워크입니다.

1. 구조
- LLM의 은닉 상태(hidden states)를 입력으로 받아 스칼라 값(보상 추정치)을 출력합니다.
- 일반적으로 간단한 MLP(다층 퍼셉트론) 구조를 가지며, LLM 위에 추가됩니다.

2. 작동 방식
- PPO 같은 강화학습 알고리즘에서 Value Head는 특정 상태의 미래 기대 보상을 예측합니다.
- 이 예측은 현재 정책의 행동이 얼마나 좋은지 평가하는 데 사용됩니다.
- LLM과 Value Head는 함께 학습되며, LLM은 정책(policy)을, Value Head는 가치 함수(value function)를 담당합니다.

3. TRL에서의 구현
   TRL 라이브러리에서는 다음과 같은 방식으로 Value Head를 구현합니다:
"""


class ValueHead(nn.Module):
    """
    Value Head 예시 구현 (TRL 라이브러리 기반 간소화 버전)
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # 첫 번째 선형 레이어
            nn.ReLU(),  # 활성화 함수
            nn.Dropout(dropout),  # 드롭아웃 (과적합 방지)
            nn.Linear(hidden_size, 1),  # 출력 레이어 (스칼라 값 출력)
        )

    def forward(self, hidden_states):
        """
        LLM의 은닉 상태를 받아 보상 값을 예측합니다.

        Args:
            hidden_states: 마지막 레이어의 은닉 상태 또는 풀링된 출력

        Returns:
            value: 예측된 가치/보상 값
        """
        # 일반적으로 마지막 토큰의 은닉 상태나 평균 풀링된 상태를 사용
        value = self.value_head(hidden_states)
        return value


"""
4. 메모리 효율성 고려사항

LLM RL이 메모리 관점에서 어려운 이유:
- LLM 자체가 이미 큰 메모리를 차지합니다.
- PPO에서는 참조 모델(reference model)도 필요해 메모리 요구량이 두 배가 됩니다.
- 강화학습은 많은 샘플링과 반복이 필요해 계산 비용이 높습니다.

메모리 효율성을 위한 해결책:
- LoRA와 같은 파라미터 효율적 미세조정(PEFT) 기법 사용
- 그래디언트 체크포인팅(gradient checkpointing) 활용
- 혼합 정밀도 학습(mixed precision training)
- 모델 병렬화(model parallelism)
"""


class AutoModelForCausalLMWithValueHead(nn.Module):
    def __init__(self, base_model, value_head):
        super().__init__()
        self.base_model = base_model  # LLM
        self.value_head = value_head  # Value Head

    def forward(self, input_ids, attention_mask=None):
        # LLM으로부터 출력 얻기
        outputs = self.base_model(input_ids, attention_mask=attention_mask)

        # 마지막 은닉 상태 가져오기
        last_hidden_state = outputs.last_hidden_state[:, -1, :]

        # Value Head를 통해 가치 예측
        value = self.value_head(last_hidden_state)

        return outputs, value


"""
PPO (Proximal Policy Optimization):
- 정책 그래디언트 방법의 일종으로, 정책과 가치 네트워크를 모두 사용합니다.
- Value Head가 필요하며, 참조 모델도 필요해 메모리 요구량이 큽니다.
- 명시적인 보상 함수가 필요합니다.

DPO (Direct Preference Optimization):
- 선호도 데이터에서 직접 정책을 최적화합니다.
- 별도의 보상 모델이나 Value Head가 필요 없어 메모리 효율적입니다.
- 인간 선호도 데이터(chosen vs rejected)를 직접 사용합니다.
"""
