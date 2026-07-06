"""Streamlit app for TCGA cancer prediction and gene-evidence RAG explanation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from src.data_loader import PROJECT_ROOT
from src.display import DEMO_NAME_NOTICE, build_patient_names, cancer_label, sample_option_label
from src.llm.chat import generate_user_explanation, hybrid_answer, overall_evidence_status, summarize_prediction
from src.llm.providers import api_available, selected_provider
from src.predict import available_sample_ids, predict
from src.rag.retrieve import retrieve_gene_evidence

RECENT_CHAT_TURNS = 4

EVIDENCE_MODE_KO = {
    "local": "로컬 근거",
    "web": "외부 검색 근거 (검증 필요)",
    "mixed": "로컬 + 외부 근거",
    "none": "근거 제한적",
}

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TOP_GENES_PATH = PROJECT_ROOT / "reports" / "top_genes_by_model.csv"
LABEL_COLUMN = "label"

MODEL_KO = {
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "mlp": "MLP",
}

STATUS_KO = {
    "curated": "근거 정리 완료",
    "draft": "근거 정리 중",
    "none": "근거 문서 없음",
}
STATUS_RANK = {"curated": 0, "draft": 1, "none": 2}
SECTION_KO = {
    "summary": "유전자 기능 요약",
    "cancer_relevance": "암종 관련 근거",
    "pathway": "관련 경로/생물학적 기능",
    "therapeutic_relevance": "치료 표적 관련성",
    "sources": "출처",
    "evidence_limitations": "근거 한계",
}
TOP_GENES_SHOWN = 10

st.set_page_config(page_title="TCGA 암종 예측 및 유전자 근거 설명", layout="wide")


@st.cache_data(show_spinner=False)
def get_sample_ids() -> list[str]:
    return list(available_sample_ids())


@st.cache_data(show_spinner=False)
def get_true_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    for split in ("train.parquet", "test.parquet"):
        path = PROCESSED_DIR / split
        if path.exists():
            series = pd.read_parquet(path, columns=[LABEL_COLUMN])[LABEL_COLUMN]
            labels.update(series.to_dict())
    return labels


@st.cache_data(show_spinner=True)
def run_prediction(sample_id: str) -> dict:
    return predict(sample_id)


@st.cache_data(show_spinner=False)
def get_cancer_top_genes(cancer_code: str, top_n: int = TOP_GENES_SHOWN) -> list[dict]:
    frame = pd.read_csv(TOP_GENES_PATH)
    per_class = frame[(frame["scope"] == "per_class") & (frame["cancer_type"] == cancer_code)]
    agg: dict[str, dict] = {}
    for row in per_class.itertuples(index=False):
        entry = agg.setdefault(row.gene, {"models": set(), "best_rank": row.rank})
        entry["models"].add(row.model)
        entry["best_rank"] = min(entry["best_rank"], row.rank)

    rows: list[dict] = []
    for gene, entry in agg.items():
        ev = retrieve_gene_evidence(gene, cancer_code)
        status = "curated" if ev["has_curated_evidence"] else ("draft" if ev["found"] else "none")
        rows.append(
            {
                "gene": gene,
                "models": sorted(entry["models"]),
                "n_models": len(entry["models"]),
                "best_rank": entry["best_rank"],
                "evidence_status": status,
            }
        )
    rows.sort(key=lambda r: (STATUS_RANK[r["evidence_status"]], r["best_rank"], r["gene"]))
    return rows[:top_n]


st.title("TCGA 암종 예측 및 유전자 근거 설명 시스템")
st.error(
    "**연구 / 교육 목적 전용입니다. 임상 진단 도구가 아닙니다.** "
    "예측과 설명은 참고용이며 실제 진단·치료 결정에 사용할 수 없습니다."
)

sample_ids = get_sample_ids()
true_labels = get_true_labels()
patient_names = build_patient_names(sample_ids)

with st.sidebar:
    st.header("샘플 선택")
    sample_id = st.selectbox(
        "환자(샘플)를 선택하세요",
        sample_ids,
        index=0,
        format_func=lambda sid: sample_option_label(sid, patient_names, true_labels),
    )
    true_label = true_labels.get(sample_id, "unknown")
    st.caption(f"{patient_names[sample_id]} ({sample_id})")
    st.metric("실제 암종", cancer_label(true_label))
    st.info(DEMO_NAME_NOTICE)

result = run_prediction(sample_id)
predictions = result["predictions"]
summary = summarize_prediction(predictions)
cancer_code = summary["final_label"]
cancer_name = cancer_label(cancer_code)
cancer_genes = get_cancer_top_genes(cancer_code)
evidence_status = overall_evidence_status(cancer_genes)
context = {
    "sample_id": sample_id,
    "cancer_code": cancer_code,
    "cancer_name": cancer_name,
    "summary": summary,
    "cancer_genes": cancer_genes,
    "evidence_status": evidence_status,
}
top_candidate_names = ", ".join(g["gene"] for g in cancer_genes[:3]) or "-"

st.subheader("예측 요약")
c1, c2, c3, c4 = st.columns(4)
c1.metric("예측 암종", cancer_name)
c2.metric("모델 동의", f"{summary['n_agree']} / {summary['n_models']}")
c3.metric("주요 유전자 후보", top_candidate_names)
c4.metric("근거 상태", evidence_status)

if true_label != "unknown":
    if cancer_code == true_label:
        st.success(f"예측 {cancer_name}은 실제 암종 {cancer_label(true_label)}과 일치합니다.")
    else:
        st.warning(f"예측 {cancer_name}은 실제 암종 {cancer_label(true_label)}과 다릅니다.")

tab_pred, tab_chat = st.tabs(["예측 & 유전자 근거", "AI에게 질문하기"])

with tab_pred:
    st.markdown("#### AI 설명")
    st.info(generate_user_explanation(context))

    st.markdown("#### 모델별 예측 결과")
    rows = []
    for name, out in predictions.items():
        top_prob = max(out["probabilities"].values())
        rows.append(
            {
                "모델": MODEL_KO.get(name, name),
                "예측 암종": cancer_label(out["predicted_label"]),
                "최고 확률": f"{top_prob:.1%}",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown(f"#### {cancer_name} 관련 유전자 후보")
    st.caption(
        "모델이 해당 암종을 구분할 때 중요하게 본 유전자 후보입니다. "
        "원인 유전자로 단정하지 않고, 근거 문서가 있는 후보를 우선 표시합니다."
    )
    if cancer_genes:
        gene_df = pd.DataFrame(
            {
                "유전자": [g["gene"] for g in cancer_genes],
                "지목 모델": [", ".join(MODEL_KO.get(m, m) for m in g["models"]) for g in cancer_genes],
                "근거 상태": [STATUS_KO[g["evidence_status"]] for g in cancer_genes],
            }
        )
        st.dataframe(gene_df, width="stretch", hide_index=True)
    else:
        st.write("이 암종에 대한 per-class 유전자 후보가 없습니다.")

    st.markdown("#### 유전자 근거 문서")
    gene_choices = [g["gene"] for g in cancer_genes]
    if gene_choices:
        selected_gene = st.selectbox("근거를 확인할 유전자를 선택하세요", gene_choices)
        evidence = retrieve_gene_evidence(selected_gene, cancer_code, web_fallback=True)
        state = evidence.get("evidence_state")
        if state == "local_curated":
            st.success(f"{evidence['gene']} 로컬 근거 문서가 있습니다.")
            for name, label in SECTION_KO.items():
                content = evidence["sections"].get(name, "").strip()
                if content:
                    with st.expander(label, expanded=(name in ("summary", "cancer_relevance"))):
                        st.write(content)
        elif state == "external_review":
            web = evidence["web"]
            local_note = "로컬 문서는 작성 중입니다." if evidence["found"] else "로컬 근거 문서가 없습니다."
            st.warning(f"{selected_gene}: {local_note} 외부 출처 검토가 필요합니다.")
            if len(web.get("aliases_searched", [])) > 1:
                st.caption(f"함께 검색한 별칭: {', '.join(web['aliases_searched'])}")
            if web.get("external_summary"):
                es = web["external_summary"]
                st.markdown(f"**외부 요약 ({es['source']}, 검증 필요):** {es['text']}")
            st.markdown("**외부 출처 (검증 필요):**")
            for link in web.get("sources", []):
                st.markdown(f"- [{link['source']} - {link['queried_symbol']}]({link['url']})")
        else:
            st.error(f"{selected_gene}: 현재 근거가 제한적입니다.")


def _render_assistant(payload: dict) -> None:
    st.markdown(payload["answer"])
    mode = EVIDENCE_MODE_KO.get(payload["evidence_mode"], payload["evidence_mode"])
    st.caption(f"답변 생성: **{payload['provider_label']}** · 근거 유형: {mode}")
    graph_edges = payload.get("graph_edges") or []
    if payload.get("graph_available"):
        st.caption(f"GraphRAG: 연결됨 · 관계 {len(graph_edges)}개")
    else:
        st.caption("GraphRAG: 비활성 또는 연결 실패")
    if payload["provider"] is None:
        ollama_failed = any(a[0] == "ollama" for a in payload.get("attempts", []))
        if ollama_failed:
            st.caption("로컬 LLM 응답 실패, 문서 기반 답변으로 전환했습니다.")
        if payload.get("fallback_reason"):
            st.caption(f"사유: {payload['fallback_reason'][:120]}")
    if payload["sources"]:
        with st.expander(f"근거·출처 {len(payload['sources'])}건 보기"):
            for src in payload["sources"][:10]:
                if src["source_url"].startswith("http"):
                    st.markdown(f"- ({src['type']}) [{src['label']}]({src['source_url']})")
                else:
                    st.markdown(f"- ({src['type']}) {src['label']}: `{src['source_url']}`")


with tab_chat:
    if api_available():
        st.success("답변 생성 provider: Gemini API 사용 가능 (실패 시 자동 fallback).")
    elif selected_provider() == "ollama":
        st.info("provider=ollama 설정: 로컬 Gemma를 시도하고 실패 시 문서 기반 답변으로 전환합니다.")
    else:
        st.info(
            "GOOGLE_API_KEY 미설정: 로컬 Gemma 또는 문서 기반으로 답변합니다. "
            "환경변수 GOOGLE_API_KEY를 설정하면 Gemini API로 더 자연스러운 답변을 생성합니다."
        )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    curated_genes = [g["gene"] for g in cancer_genes if g["evidence_status"] == "curated"]
    gene_example = (
        f"{curated_genes[0]} 유전자는 어떤 근거가 있어?"
        if curated_genes
        else (f"{cancer_genes[0]['gene']}는 어떤 근거가 있어?" if cancer_genes else "어떤 유전자가 중요했어?")
    )
    example_questions = [
        f"왜 이 샘플은 {cancer_name}으로 예측됐어?",
        "어떤 유전자가 중요했어?",
        gene_example,
        "이 유전자가 원인이야, 아니면 관련 신호야?",
        "근거가 충분해?",
        "LASS3는 근거가 부족하다는데 외부 자료도 찾아줘",
    ]

    st.caption("추천 질문 (클릭하면 바로 답변):")
    pending_question = None
    for row_start in range(0, len(example_questions), 2):
        cols = st.columns(2)
        for col, example in zip(cols, example_questions[row_start : row_start + 2]):
            if col.button(example, key=f"ex_{example}", use_container_width=True):
                pending_question = example

    submitted_question = None
    left, right = st.columns([1, 1.5])
    with left:
        st.markdown("##### 현재 샘플")
        st.write(f"예측 암종: **{cancer_name}**")
        st.write(f"모델 동의: {summary['n_agree']} / {summary['n_models']}")
        st.write(f"근거 상태: **{evidence_status}**")
        st.markdown("**주요 유전자 후보**")
        for gene in cancer_genes[:5]:
            st.write(f"- {gene['gene']} · {STATUS_KO[gene['evidence_status']]}")

    with right:
        head_col, btn_col = st.columns([3, 1])
        head_col.markdown("##### RAG 대화")
        if btn_col.button("초기화", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

        with st.form("rag_question_form", clear_on_submit=True):
            typed_question = st.text_input(
                "질문 입력",
                placeholder="예: TG는 THCA와 어떻게 연결돼 있어?",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("질문하기", use_container_width=True)
        if submitted and typed_question.strip():
            submitted_question = typed_question.strip()

        messages = st.session_state.chat_messages
        recent = messages[-(RECENT_CHAT_TURNS * 2) :]
        with st.container(height=340):
            if not recent:
                st.caption("질문을 입력하거나 추천 질문을 눌러보세요.")
            for role, payload in recent:
                with st.chat_message(role):
                    if role == "user":
                        st.markdown(payload)
                    else:
                        _render_assistant(payload)

        if len(messages) > len(recent):
            with st.expander(f"전체 대화 보기 ({len(messages) // 2}턴)"):
                for role, payload in messages:
                    with st.chat_message(role):
                        if role == "user":
                            st.markdown(payload)
                        else:
                            _render_assistant(payload)

    user_question = submitted_question or pending_question
    if user_question:
        with st.spinner("근거를 검색하고 답변을 생성하는 중..."):
            res = hybrid_answer(user_question, summary, cancer_code, cancer_name, cancer_genes)
        st.session_state.chat_messages.append(("user", user_question))
        st.session_state.chat_messages.append(("assistant", res))
        st.rerun()
