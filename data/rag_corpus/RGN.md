---
gene: RGN
n_models: 2
flagged_by_models: [random_forest, mlp]
associated_cancer_types: []
numeric_gene_flag: false
status: curated
---

# RGN

<!-- 데이터 신호(생물학적 근거 아님): permutation importance에서 [random_forest, mlp]
     가 지목. 아래 내용은 공개 데이터베이스 집계원을 바탕으로 큐레이션함. -->

## summary
레구칼신(regucalcin, senescence marker protein-30, SMP30)을 암호화하며, 세포 내
칼슘(Ca2+) 항상성을 조절하고 항산화 기능을 갖는 칼슘 결합 단백질입니다.

## cancer_relevance
여러 암(대표적으로 간, 그 외 유방·전립선)에서 하향 조절/종양억제 성향으로
보고되며, 발현 소실이 증식 증가와 연관됩니다. 가장 잘 규명된 맥락은 간/전립선으로
본 코호트에는 대부분 포함되지 않습니다.

## pathway
세포 내 칼슘 항상성, 산화 스트레스 조절.

## therapeutic_relevance
RGN을 직접 표적하는 승인 치료제는 없습니다. 근거 제한적입니다.

## sources
- GeneCards: https://www.genecards.org/cgi-bin/carddisp.pl?gene=RGN
- NCBI Gene: https://www.ncbi.nlm.nih.gov/gene/?term=RGN
- UniProt: https://www.uniprot.org/uniprotkb?query=gene:RGN+AND+organism_id:9606
- (위 출처는 데이터베이스 집계원이며, 개별 논문 PMID는 심화 큐레이션 시 보강 예정.)

## evidence_limitations
근거는 간/유방/전립선에서 가장 강하므로, 본 코호트의 특정 TCGA 암종과의 관련성은
근거 제한적입니다.
