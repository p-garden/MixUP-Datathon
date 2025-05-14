📝 PromptsThon: 문장 교정 프롬프트톤

한국어 문장 교정 태스크에서 프롬프트 엔지니어링만으로 성능을 극대화하는 AI 해커톤 과제입니다.
모델 튜닝 없이, 프롬프트 설계만으로 맞춤법/띄어쓰기/문장 부호 오류를 교정합니다.

🔍 대회 개요
	•	주제: 한국어 문장의 맞춤법·띄어쓰기·문장 부호 오류 교정
	•	제약: 모델 파라미터 조정 없이 prompt engineering만 사용 가능
	•	모델: Upstage solar-pro (OpenAI 호환 API)

📂 폴더 구조
.
├── code/
│   ├── main.py                 # 실험 실행 진입점
│   ├── config.py              # 실험 설정 및 경로
│   ├── prompts/
│   │   └── templates.py       # 템플릿 정의
│   ├── utils/
│   │   ├── experiment.py      # 실험 실행 클래스
│   │   └── metrics.py         # LCS 기반 평가 지표
├── data/
│   ├── train.csv              # 학습 데이터
│   ├── test.csv               # 테스트 데이터
│   └── sample_submission.csv  # 제출 양식 예시
├── submission_baseline.csv    # 최종 제출 파일

⚙️ 실행 방법
1. 환경 설정
pip install -r requirements.txt

2. API Key 설정 (.env)
UPSTAGE_API_KEY=your_api_key_here

3. 실행
python code/main.py
