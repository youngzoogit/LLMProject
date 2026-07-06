---
gene: SFTPB
n_models: 2
flagged_by_models: [random_forest, mlp]
associated_cancer_types: [LUAD, LUSC]
numeric_gene_flag: false
status: curated
---

# SFTPB

<!-- 데이터 신호(생물학적 근거 아님): permutation importance에서 [random_forest, mlp]
     가 지목. 아래 내용은 공개 데이터베이스 집계원을 바탕으로 큐레이션함. -->

## summary
폐 계면활성단백 B(SP-B)를 암호화하며, 계면활성제 기능과 폐포 안정에 필수적인
소수성 분비 단백질입니다. 폐 II형 폐포세포에서 발현이 강하게 나타납니다.

## cancer_relevance
폐/폐포(II형 폐세포) 분화의 마커로 널리 쓰이며 폐선암에서 발현됩니다. SFTPB 신호가
높으면 본 코호트에서 폐 기원 종양(LUAD/LUSC)일 가능성과 부합합니다. 다만 예후
마커로서의 가치는 맥락에 따라 다릅니다.

## pathway
폐 계면활성제 대사/항상성, 지질 결합, lamellar body 분비.

## therapeutic_relevance
SFTPB를 직접 표적하는 승인 치료제는 없으며, 주로 계통/분화 마커로 사용됩니다.
표적치료 관련 주장은 근거 제한적입니다.

## sources
- GeneCards: https://www.genecards.org/cgi-bin/carddisp.pl?gene=SFTPB
- NCBI Gene: https://www.ncbi.nlm.nih.gov/gene/?term=SFTPB
- UniProt: https://www.uniprot.org/uniprotkb?query=gene:SFTPB+AND+organism_id:9606
- (위 출처는 데이터베이스 집계원이며, 개별 논문 PMID는 심화 큐레이션 시 보강 예정.)

## evidence_limitations
폐 계통 마커 역할은 잘 확립되어 있습니다. 그러나 TCGA LUAD/LUSC 내에서의 구체적
예후 방향성은 여기서 단정하지 않으며(근거 제한적), 해석 전 원문 문헌 확인이
필요합니다.
