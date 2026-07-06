---
gene: LUM
n_models: 2
flagged_by_models: [random_forest, mlp]
associated_cancer_types: [BRCA, COAD]
numeric_gene_flag: false
status: curated
---

# LUM

<!-- 데이터 신호(생물학적 근거 아님): permutation importance에서 [random_forest, mlp]
     가 지목. 아래 내용은 공개 데이터베이스 집계원을 바탕으로 큐레이션함. -->

## summary
루미칸(lumican)을 암호화하며, 콜라겐과 결합해 콜라겐 섬유 조립을 조절하는
세포외기질(ECM)의 class II 소형 류신 풍부 프로테오글리칸(SLRP)입니다.

## cancer_relevance
종양-기질 상호작용에 관여하는 기질/ECM 성분으로, 유방암과 대장암 등에서 연구되어
왔습니다. 보고된 역할은 맥락 의존적입니다(종양 억제/촉진 양쪽 모두 보고).

## pathway
세포외기질 구성, 콜라겐 섬유 형성, TGF-beta 및 integrin 신호 조절.

## therapeutic_relevance
LUM을 직접 표적하는 승인 치료제는 없습니다. 근거 제한적입니다.

## sources
- GeneCards: https://www.genecards.org/cgi-bin/carddisp.pl?gene=LUM
- NCBI Gene: https://www.ncbi.nlm.nih.gov/gene/?term=LUM
- UniProt: https://www.uniprot.org/uniprotkb?query=gene:LUM+AND+organism_id:9606
- (위 출처는 데이터베이스 집계원이며, 개별 논문 PMID는 심화 큐레이션 시 보강 예정.)

## evidence_limitations
효과의 방향성이 맥락 의존적이고 본 코호트의 특정 TCGA 암종에서는 확정되지
않았으므로, 암종별 주장은 근거 제한적으로 다루어야 합니다.
