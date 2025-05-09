import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    user_prompt = """
1부터 8까지의 숫자를 원형으로 나열하려고 합니다.
나열된 순서대로(그리고 마지막과 첫 번째 사이에도) 인접한 두 숫자의 합이 항상 소수여야 합니다.

예를 들어 …, 3 — 8 … 처럼 인접하면 3+8=11(소수)이 되어야 하고,
끝과 처음도 “… 4 — 1 …” 처럼 4+1=5(소수)가 돼야 합니다.

문제.
이 조건을 만족하는 서로 다른 순환 배열(회전으로 같은 건 하나로 칩니다)은 총 몇 개일까요?
서로 다른 순환 배열의 개수와 해를 모두 구하세요.
"""

    response = get_response(user_prompt)
    print(response)
