from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import re

load_dotenv()  # .env 파일 로드

gpt_api_key = os.getenv("GPT_API_KEY")
client = OpenAI(api_key=gpt_api_key)

stt_example = "안녕하세요. 오늘 저는 AI 기술의 발전이 우리의 일상에 어떤 변화를 가져오고 있는지에 대해 이야기해 보려 합니다. 불과 몇 년 전만 해도, 인공지능은 알파거나 로브처럼 특정 한 분야에서만 존재하는 기술로 여유졌습니다. 하지만 지금은 어떨까요? 우리가 매일 사용하는 스마트폰의 음성비서, 유튜브나 넷플릭스의 추천 시스템, 심지어 스마트홈까지 AI는 이미 우리 삶 곳곳에 자연스럽게 스며들어 있습니다. 특히, 채 GPT 같은 생성형 AI는 텍스트 요약, 이메일 작성, 코드 디머깅 등 실질적인 업무에 도와줄 수 있는 도구로 각광받고 있습니다. 많은 회사들이 AI를 업무 효율화에 적극적으로 사용하고 있고, 교육 분야에서도 학생 개인 맞춤형 피드백을 제공한 등 다양한 시도가 이루어지고 있습니다. 하지만 이처럼 편리한 기술 뒤에는 분명히 고민거리도 존재합니다. 개인정보 보호 문제, 일자리 대체에 대한 우려, 그리고 AI의 윤리적 판단 문제 등은 앞으로 우리가 반드시 풀어야 할 과제입니다. 결론적으로 AI는 이제 특별한 기술이 아닌 우리 삶의 일부로 자리 잡았습니다. 중요한 것은 이 기술을 어떻게 활용하고 어떤 기준과 방향성을 가지고 발전시켜 나갈 것인가 하는 점입니다. 이상으로 발표 마치겠습니다. 감사합니다."

## TODO 단어, 문맥 수정

stt_edit_prompt = """
다음 문장은 STT(음성 인식) 결과야.  
잘못 인식된 단어만 고쳐줘.  
말투, 어순, 표현 방식은 바꾸지 말고 가능한 한 그대로 유지해.  
예상되는 오류: 유사 발음으로 인한 단어 대체, 조사 누락, 단어 생략 등
### 입력 문장
{stt_result}
### 출력 형식
```json
{{
  "corrected_sentence": "수정된 문장"
}}
```
"""
def correct_stt_result(stt_result: str) -> str:
    prompt = stt_edit_prompt.format(stt_result=stt_result)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r"```json\n|```", "", content).strip()

    try:
        corrected = json.loads(content)["corrected_sentence"]
        return corrected
    except (json.JSONDecodeError, KeyError):
        return "[오류] corrected_sentence를 추출하지 못했습니다."

analysis_prompt = """
너는 발표 대본을 조정하고 분석하는 LLM 기반 발표 어시스턴트야.  
아래 작업을 수행한 후, 결과를 반드시 JSON 형식으로 출력해줘.

### 작업
1. 발표 대본을 자연스러운 발표체로 수정한다.  
   - 문장 구조, 어휘, 표현 등을 자연스럽게 조정

2. 발표 대본의 길이를 조정한다.  
   - 현재 발표 시간: {current_time}
   - 목표 발표 시간: {target_time}
   - 시간이 길면 요약, 짧으면 내용 보완  
   - 자연스러운 발표체 유지

3. 발표 대본에 대해 다음 피드백을 제공한다:  
   - 자주 반복되는 단어 5개  
   - 어색하거나 부자연스러운 문장 및 수정 예시  
   - 너무 어려운 또는 쉬운 표현 지적

4. 발표 내용 기반 예상 질문을 3~5개 생성한다.

### 입력 대본
{corrected_stt_result}

### 출력 형식 (JSON)
```json
{{
  "adjusted_script": "수정된 발표 대본...",
  "feedback": {{
    "frequent_words": ["단어1", "단어2", "단어3", "단어4", "단어5"],
    "awkward_sentences": [
      {{
        "original": "어색한 문장 예시",
        "suggestion": "자연스러운 수정 예시"
      }}
    ],
    "difficulty_issues": [
      {{
        "type": "too_easy",
        "sentence": "너무 쉬운 문장",
        "suggestion": "조금 더 전문적인 표현"
      }},
      {{
        "type": "too_difficult",
        "sentence": "너무 어려운 문장",
        "suggestion": "쉽게 바꾼 표현"
      }}
    ]
  }},
  "predicted_questions": [
    "예상 질문 1",
    "예상 질문 2",
    "예상 질문 3"
  ]
}}
"""

def get_chat_response(corrected_stt_result, current_time="6:00", target_time="6:00"):
    prompt = analysis_prompt.format(
        corrected_stt_result=corrected_stt_result,
        current_time=current_time,
        target_time=target_time
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r"```json\n?|```", "", content).strip()  # 코드 블록 제거

    try:
        parsed = json.loads(content)
        return {
            "adjusted_script": parsed.get("adjusted_script"),
            "feedback": parsed.get("feedback"),
            "predicted_questions": parsed.get("predicted_questions")
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[오류] 분석 결과 파싱 실패: {e}")
        return {
            "adjusted_script": None,
            "feedback": None,
            "predicted_questions": None
        }


# print(stt_example)
# co_stt = correct_stt_result(stt_example)
# print(co_stt)
# adjusted_script, feedback, predicted_questions = get_chat_response(co_stt, "2:30", "3:00")
# print(adjusted_script)
# print(feedback["frequent_words"])
# print(feedback["awkward_sentences"][0]["original"])
# print(feedback["difficulty_issues"][1]["suggestion"])
# print(predicted_questions)