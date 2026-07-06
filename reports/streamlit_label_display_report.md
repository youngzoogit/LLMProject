# Streamlit 표시 이름 개선 보고서

_목표: 내부 TCGA `sample_id` 와 암종 코드는 그대로 유지하되, 화면에서는 익명화된
데모 환자명과 한국어 암종명을 함께 보여준다._

## 1. 신규 표시 모듈 — `src/display.py`
- `CANCER_KO`: 암종 코드 -> 한국어명 매핑
  (BRCA 유방암, LUAD 폐선암, LUSC 폐편평세포암, KIRC 신장 투명세포암,
  COAD 대장암, THCA 갑상선암).
- `cancer_label(code)` -> `"갑상선암 (THCA)"` 형태(미지정 코드는 원본 코드로 폴백).
- `build_patient_names(sample_ids)` -> index 순서로 `환자 A-001`, `환자 A-002`, ...
  자동 생성(999개마다 블록 문자 A->B->C, 3,604개까지 A~D). 수동 작성 불필요.
- `sample_option_label(...)` -> `"환자 A-001 | 실제 암종: 갑상선암 (THCA)"`.
- `DEMO_NAME_NOTICE`: "환자명은 실제 개인정보가 아니라 ... 익명화된 데모 이름입니다."

내부 예측은 여전히 `predict(sample_id)` 로 실제 `sample_id` 를 사용합니다.
표시 계층만 변경했습니다.

## 2. 화면 반영 (`app.py`)
- **샘플 선택 selectbox**: 값은 `sample_id`(내부용) 유지, `format_func` 로 화면에는
  `환자 A-001 | 실제 암종: 갑상선암 (THCA)` 표시.
- **사이드바**: 선택 시 `환자 A-001 (TCGA-KS-A4I5-01)` 병기, "실제 암종"을
  `갑상선암 (THCA)` 로 표시, 하단에 익명화 안내 문구(ℹ️) 노출.
- **요약 카드**: "최종 예측 암종"을 `갑상선암 (THCA)` 로 표시, 정답 일치 메시지도
  한국어 암종명으로.
- **모델별 예측 표**: "예측 암종" 컬럼을 `갑상선암 (THCA)` 형식으로 표시.
- **AI 설명 / RAG Chat**: `src/llm/chat.py` 가 `cancer_label` 을 사용해 설명과 답변에
  한국어 암종명을 함께 노출.

## 3. 수정/신규 파일
- 신규: `src/display.py`
- 수정: `app.py` (selectbox/사이드바/요약카드/모델표에 표시 헬퍼 적용)
- 수정: `src/llm/chat.py` (설명/답변에 `cancer_label` 적용)

## 4. 검증 결과
- 단위 확인:
  - `build_patient_names` -> `환자 A-001, 환자 A-002, 환자 A-003 ...`
  - `cancer_label("THCA")` -> `갑상선암 (THCA)`
  - `sample_option_label(...)` -> `환자 A-001 | 실제 암종: 갑상선암 (THCA)`
- Streamlit `AppTest`:
  - **예외 0개 (CLEAN)**.
  - selectbox 옵션 라벨에 "환자" + "실제 암종" 포함 확인.
  - 화면 텍스트에 한국어 암종명 및 `(THCA)` 등 코드 병기 형식 존재 확인.
  - 모델별 예측 표의 "예측 암종" 값 = `['갑상선암 (THCA)', '갑상선암 (THCA)',
    '갑상선암 (THCA)']` (한국어명 + 코드 병기).

## 5. 참고
- 환자명은 표시 순서(현재 train+test 병합 순서) 기준으로 결정론적으로 생성됩니다.
  샘플 순서가 바뀌면 번호도 바뀌므로, 고정 매핑이 필요하면 `sample_id -> 이름` 을
  파일로 저장하는 방식으로 확장할 수 있습니다.
- 암종 코드가 6종 외로 늘어나면 `CANCER_KO` 에 항목만 추가하면 됩니다.
