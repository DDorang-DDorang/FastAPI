from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import re

load_dotenv()  # .env 파일 로드

gpt_api_key = os.getenv("GPT_API_KEY")
client = OpenAI(api_key=gpt_api_key)



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
   - 어색한 문장과 수정 
   - 쉬운 표현, 어려운 표현 및 개선안
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

compare_prompt = """
당신은 발표 코칭 전문가입니다.  
사용자가 두 개의 발표 대본(script1, script2)을 입력하면, 
두 번째 대본(script2)이 첫 번째 대본(script1)에 비해 어떻게 발전했는지 분석하고,
아직 개선해야 할 부분을 중심으로 피드백을 제공합니다.  

단, 결과를 작성할 때는 'script1', 'script2'라는 표현을 사용하지 말고  
대신 '이전 발표', '이번 발표'라는 자연스러운 표현을 사용하세요.  

발표의 성격에 따라 청중 참여나 상호작용은 필수 요소가 아닙니다.
설득형 발표라면 청중 참여를 권장할 수 있으나, 정보 전달형 발표라면 명확한 구조와 논리적 흐름을 더 중점적으로 평가하세요

출력은 반드시 아래 JSON 형식을 따르세요.  
불필요한 설명이나 서두는 포함하지 마세요.  

입력:
{{
  "script1": "{script1}",
  "script2": "{script2}"
}}

출력 형식 예시:
{{
  "improvements_made": "이번 발표가 이전 발표에 비해 발전한 점",
  "areas_to_improve": "이번 발표에서 여전히 보완이 필요한 부분",
  "overall_feedback": "이번 발표에 대한 종합 평가"
}}

평가 기준:
- 발표 구조의 완성도 (도입, 전개, 결론의 논리적 연결)
- 내용의 구체성과 깊이 (핵심 주제에 대한 이해, 예시의 적절성)
- 논리적 흐름과 설득력 (근거의 타당성, 메시지의 일관성)
- 언어 표현의 명확성과 자연스러움
- 전달력과 자신감 
- 전체적인 완성도와 전문성
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
    
def get_compare_result(script1, script2) :
    prompt = compare_prompt.format(
        script1 = script1,
        script2 = script2
    )
    content = call_gpt(prompt, temperature=0)
    try:
        return json.loads(content)
    except (json.JSONDecodeError, KeyError):
        return {
            "strengths_comparison": None,
            "improvement_suggestions": None,
            "overall_feedback": None
        }



if __name__ == "__main__":
    #stt_example = "안녕하세요. 오늘 저는 AI 기술의 발전이 우리의 일상에 어떤 변화를 가져오고 있는지에 대해 이야기해 보려 합니다. 불과 몇 년 전만 해도, 인공지능은 알파거나 로브처럼 특정 한 분야에서만 존재하는 기술로 여유졌습니다. 하지만 지금은 어떨까요? 우리가 매일 사용하는 스마트폰의 음성비서, 유튜브나 넷플릭스의 추천 시스템, 심지어 스마트홈까지 AI는 이미 우리 삶 곳곳에 자연스럽게 스며들어 있습니다. 특히, 채 GPT 같은 생성형 AI는 텍스트 요약, 이메일 작성, 코드 디머깅 등 실질적인 업무에 도와줄 수 있는 도구로 각광받고 있습니다. 많은 회사들이 AI를 업무 효율화에 적극적으로 사용하고 있고, 교육 분야에서도 학생 개인 맞춤형 피드백을 제공한 등 다양한 시도가 이루어지고 있습니다. 하지만 이처럼 편리한 기술 뒤에는 분명히 고민거리도 존재합니다. 개인정보 보호 문제, 일자리 대체에 대한 우려, 그리고 AI의 윤리적 판단 문제 등은 앞으로 우리가 반드시 풀어야 할 과제입니다. 결론적으로 AI는 이제 특별한 기술이 아닌 우리 삶의 일부로 자리 잡았습니다. 중요한 것은 이 기술을 어떻게 활용하고 어떤 기준과 방향성을 가지고 발전시켜 나갈 것인가 하는 점입니다. 이상으로 발표 마치겠습니다. 감사합니다."
    #print(stt_example)
    #print()
    #co_stt = correct_stt_result(stt_example)
    #print(co_stt)
    #print()
    #result = get_chat_response(co_stt, "1:30", "3:00")
    #print(result["adjusted_script"])
    #print(result["feedback"])
    #print(result["predicted_questions"])

    worse_script = """  안녕하세요.
                        오늘 저는 AI가 교육에 어떤 영향을 주는가에 대해 이야기하겠습니다.

                        요즘 인공지능 기술이 빠르게 발전하면서, 교육에서도 AI가 많이 사용되고 있습니다.
                        AI는 학생들에게 도움이 되는 기술이며, 앞으로 교육에 큰 변화를 줄 것입니다.

                        먼저, AI는 학습을 더 효율적으로 만듭니다.
                        AI는 학생들의 데이터를 분석해서 더 좋은 학습 방법을 제공합니다.
                        AI를 통해 학생들이 스스로 공부할 수 있고, 교사도 학생을 더 잘 도울 수 있습니다.
                        이처럼 AI는 교육에서 점점 더 중요한 역할을 하고 있습니다.

                        또한, AI는 교사에게도 도움이 됩니다.
                        AI는 학생의 성적과 태도를 분석해 교사가 수업을 더 잘 진행할 수 있게 도와줍니다.
                        이로 인해 교사는 학생을 더 잘 이해할 수 있습니다.
                        AI는 교사를 대신하지 않고, 교사를 도와주는 기술입니다.

                        AI는 교육의 기회를 넓힐 수도 있습니다.
                        온라인 학습과 결합되면 누구나 교육을 받을 수 있습니다.
                        AI는 세상을 더 나은 방향으로 바꿀 수 있는 기술입니다.

                        하지만 AI에는 문제점도 있습니다.
                        데이터가 잘못 사용될 수도 있고, 기술 의존이 심해질 수도 있습니다.
                        그래서 AI를 사용할 때는 주의가 필요합니다.

                        결론적으로, AI는 교육을 발전시키는 중요한 기술입니다.
                        앞으로 AI를 어떻게 사용하느냐에 따라 교육의 미래가 달라질 것입니다.
                        감사합니다."""

    perfect_script = """안녕하세요, 여러분.
                        오늘 저는 AI와 함께 성장하는 교육의 미래라는 주제로 이야기해보려 합니다.

                        우리는 지금, 인공지능이 일상 깊숙이 스며든 시대에 살고 있습니다.
                        스마트폰의 음성 비서부터 추천 알고리즘, 자동 번역까지 — 이미 AI는 우리의 삶 곳곳에서 조용히 도움을 주고 있죠.
                        그렇다면 교육에서는 AI가 어떤 변화를 만들어낼 수 있을까요?

                        먼저, AI는 학습의 개인화를 가능하게 합니다.
                        모든 학생이 같은 방식으로 배우지는 않습니다.
                        AI 기반 학습 플랫폼은 학생의 학습 속도와 취약한 영역을 분석해, 각자에게 맞는 콘텐츠를 추천합니다.
                        예를 들어, 한 학생이 영어 문법에는 강하지만 듣기 이해력이 약하다면, 시스템이 자동으로 듣기 중심 문제를 더 제시하죠.
                        이처럼 AI는 하나의 커리큘럼에서 개인 맞춤형 학습으로의 전환을 이끌고 있습니다.

                        두 번째로, AI는 교사의 역할을 강화합니다.
                        AI는 학생의 데이터를 분석해 학습 패턴을 알려주고, 교사는 그 결과를 토대로 더 깊이 있는 피드백을 제공합니다.
                        즉, 교사가 단순히 지식을 전달하는 사람이 아니라, 학습을 설계하고 이끄는 코치로 변화하는 것입니다.
                        AI는 교사를 대체하는 것이 아니라, 교사의 역량을 확장시키는 동반자가 됩니다.

                        마지막으로, AI는 교육의 접근성을 넓힙니다.
                        지리적 한계나 경제적 제약으로 학습 기회를 얻기 어려웠던 학생들도,
                        AI 기반 온라인 학습 시스템을 통해 양질의 교육을 받을 수 있습니다.
                        이것이야말로 기술이 가진 진정한 가치 모두를 위한 교육을 실현하는 방향이라고 생각합니다."""


    compare_result = get_compare_result(worse_script, perfect_script)
    print(compare_result["improvements_made"])
    print(compare_result["next_focus_points"])
    print(compare_result["overall_feedback"])

    # 최근 것만 하는 걸로
    # 구체적으로 피드백, 예시   