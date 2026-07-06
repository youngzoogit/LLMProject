"""User-facing display helpers for demo patient names and Korean cancer labels."""

from __future__ import annotations

CANCER_KO: dict[str, str] = {
    "BRCA": "유방암",
    "LUAD": "폐선암",
    "LUSC": "폐편평세포암",
    "KIRC": "신장 투명세포암",
    "COAD": "대장암",
    "THCA": "갑상선암",
}

_PATIENTS_PER_BLOCK = 999

DEMO_NAME_NOTICE = (
    "환자명은 실제 개인정보가 아니라 TCGA 샘플을 이해하기 쉽게 표시하기 위한 "
    "익명 데모 이름입니다."
)


def cancer_label(code: str) -> str:
    """Format a cancer code as '한국어명 (CODE)'."""
    if not code or code == "unknown":
        return "알 수 없음"
    korean = CANCER_KO.get(code)
    return f"{korean} ({code})" if korean else code


def _demo_patient_name(index: int) -> str:
    block = index // _PATIENTS_PER_BLOCK
    number = index % _PATIENTS_PER_BLOCK + 1
    letter = chr(ord("A") + block)
    return f"환자 {letter}-{number:03d}"


def build_patient_names(sample_ids: list[str]) -> dict[str, str]:
    return {sid: _demo_patient_name(i) for i, sid in enumerate(sample_ids)}


def sample_option_label(
    sample_id: str,
    patient_names: dict[str, str],
    true_labels: dict[str, str],
) -> str:
    patient = patient_names.get(sample_id, sample_id)
    true_code = true_labels.get(sample_id, "unknown")
    return f"{patient} | 실제 암종: {cancer_label(true_code)}"