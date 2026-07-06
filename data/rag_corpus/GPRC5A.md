---
gene: GPRC5A
n_models: 2
flagged_by_models: [random_forest, mlp]
associated_cancer_types: [LUAD, LUSC]
numeric_gene_flag: false
status: curated
---

# GPRC5A

<!-- 데이터 신호(생물학적 근거 아님): permutation importance에서 [random_forest, mlp]
     가 지목. 아래 내용은 공개 데이터베이스 집계원을 바탕으로 큐레이션함. -->

## summary
레티노산 유도 GPCR family C group 5 member A(RAI3/RAIG1)를 암호화하는 고아 G단백
연결 수용체로, 정상 폐 상피에서 높게 발현되며 레티노산에 의해 유도됩니다.

## cancer_relevance
폐 종양억제자로 알려져 있습니다. Gprc5a 결손 생쥐는 자발적으로 폐 종양이 생기고,
폐암에서 이 유전자가 자주 하향 조절됩니다. 본 코호트의 LUAD/LUSC와 관련됩니다.

## pathway
레티노산 신호전달, EGFR 및 NF-kB 신호 조절이 보고됨.

## therapeutic_relevance
승인된 표적치료제는 없으며, 약물 표적이라기보다 종양억제자 바이오마커로서 관심을
받습니다. 치료 관련성은 근거 제한적입니다.

## sources
- GeneCards: https://www.genecards.org/cgi-bin/carddisp.pl?gene=GPRC5A
- NCBI Gene: https://www.ncbi.nlm.nih.gov/gene/?term=GPRC5A
- UniProt: https://www.uniprot.org/uniprotkb?query=gene:GPRC5A+AND+organism_id:9606
- (위 출처는 데이터베이스 집계원이며, 개별 논문 PMID는 심화 큐레이션 시 보강 예정.)

## evidence_limitations
종양억제자 역할은 주로 폐에서 문서화되어 있어, 폐 이외의 TCGA 암종으로의 확대
해석은 근거 제한적입니다.
