from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import re

load_dotenv()  # .env 파일 로드

gpt_api_key = os.getenv("GPT_API_KEY")
client = OpenAI(api_key=gpt_api_key)

stt_example = "안녕하세요. 오늘 저는 AI 기술의 발전이 우리의 일상에 어떤 변화를 가져오고 있는지에 대해 이야기해 보려 합니다. 불과 몇 년 전만 해도, 인공지능은 알파거나 로브처럼 특정 한 분야에서만 존재하는 기술로 여유졌습니다. 하지만 지금은 어떨까요? 우리가 매일 사용하는 스마트폰의 음성비서, 유튜브나 넷플릭스의 추천 시스템, 심지어 스마트홈까지 AI는 이미 우리 삶 곳곳에 자연스럽게 스며들어 있습니다. 특히, 채 GPT 같은 생성형 AI는 텍스트 요약, 이메일 작성, 코드 디머깅 등 실질적인 업무에 도와줄 수 있는 도구로 각광받고 있습니다. 많은 회사들이 AI를 업무 효율화에 적극적으로 사용하고 있고, 교육 분야에서도 학생 개인 맞춤형 피드백을 제공한 등 다양한 시도가 이루어지고 있습니다. 하지만 이처럼 편리한 기술 뒤에는 분명히 고민거리도 존재합니다. 개인정보 보호 문제, 일자리 대체에 대한 우려, 그리고 AI의 윤리적 판단 문제 등은 앞으로 우리가 반드시 풀어야 할 과제입니다. 결론적으로 AI는 이제 특별한 기술이 아닌 우리 삶의 일부로 자리 잡았습니다. 중요한 것은 이 기술을 어떻게 활용하고 어떤 기준과 방향성을 가지고 발전시켜 나갈 것인가 하는 점입니다. 이상으로 발표 마치겠습니다. 감사합니다."

def call_gpt(prompt: str, model="gpt-4o-mini", temperature=0.0) -> str:
    """
    GPT 호출 공통 함수
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    content = response.choices[0].message.content.strip()
    # 코드 블록 제거
    content = re.sub(r"```json\n?|```", "", content).strip()
    return content

stt_edit_prompt = """
다음 문장은 STT 결과입니다.
너의 역할은 최소한의 교정만 수행하는 것입니다.

규칙:
- 잘못 인식된 단어만 수정 (유사 발음, 조사 누락, 단어 생략 등)
- 말투, 어순, 표현은 절대 변경하지 마세요
- JSON 형식으로만 출력
- 띄어쓰기/맞춤법은 필요할 때만 수정

입력 문장:
{stt_result}

출력 예시:
{{"corrected_sentence": "수정된 문장"}}
"""

def correct_stt_result(stt_result: str) -> str:
    prompt = stt_edit_prompt.format(stt_result=stt_result)
    content = call_gpt(prompt, temperature=0)
    try:
        return json.loads(content)["corrected_sentence"]
    except (json.JSONDecodeError, KeyError):
        return "[오류] corrected_sentence 추출 실패"
    
analysis_prompt = """
너는 발표 어시스턴트입니다. 교정된 발표 대본을 받고 다음 작업을 수행하세요:

1. 발표체로 자연스럽게 수정
1. 발표체로 자연스럽게 수정합니다.
2. 목표 발표 시간에 맞추어 대본 길이를 조정합니다:
   - 현재 발표 시간: {current_time} (mm:ss)
   - 목표 발표 시간: {target_time} (mm:ss)
   - ±10% 범위 내로 조정합니다.
   - 시간 조정 규칙:
     - 현재 > 목표 → 반복 문장 제거, 불필요한 부연 설명 제거, 문장 압축
     - 현재 < 목표 → 문장 길이를 반드시 늘리세요. 다음 방법으로 확장:
       * 구체적 예시 추가
       * 관련 사례·비유 삽입
       * 핵심 문장 강조 및 부연 설명 추가
       * 반복 제거 금지

3. 피드백 제공:
   - 가장 많이 반복된 단어 5개
   - 어색한 문장 2개와 수정 예시
   - 쉬운 표현 1개, 어려운 표현 1개와 개선안
4. 예상 질문 3~5개 생성

JSON 형식으로 출력 (다른 텍스트 금지):
{{
  "adjusted_script": "...",
  "feedback": {{
    "frequent_words": ["", "", "", "", ""],
    "awkward_sentences": [
        {{"original": "...", "suggestion": "..."}},
        {{"original": "...", "suggestion": "..."}}
    ],
    "difficulty_issues": [
        {{"type": "too_easy", "sentence": "...", "suggestion": "..."}},
        {{"type": "too_difficult", "sentence": "...", "suggestion": "..."}}
    ]
  }},
  "predicted_questions": ["", "", ""]
}}

입력 문장:
{corrected_stt_result}
"""

def get_chat_response(corrected_stt_result, current_time="6:00", target_time="6:00"):
    prompt = analysis_prompt.format(
        corrected_stt_result=corrected_stt_result,
        current_time=current_time,
        target_time=target_time
    )
    content = call_gpt(prompt, temperature=0)
    try:
        return json.loads(content)
    except (json.JSONDecodeError, KeyError):
        return {
            "adjusted_script": None,
            "feedback": None,
            "predicted_questions": None
        }


print(stt_example)
print()
co_stt = correct_stt_result(stt_example)
print(co_stt)
print()
result = get_chat_response(co_stt, "1:30", "3:00")
print(result["adjusted_script"])
print(result["feedback"])
print(result["predicted_questions"])