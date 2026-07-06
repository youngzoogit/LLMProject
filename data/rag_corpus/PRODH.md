---
gene: PRODH
n_models: 2
flagged_by_models: [random_forest, mlp]
associated_cancer_types: []
numeric_gene_flag: false
status: curated
---

# PRODH

<!-- 데이터 신호(생물학적 근거 아님): permutation importance에서 [random_forest, mlp]
     가 지목. 아래 내용은 공개 데이터베이스 집계원을 바탕으로 큐레이션함. -->

## summary
프롤린 탈수소효소(proline oxidase, POX)를 암호화하며, 프롤린 이화작용의 첫 단계를
촉매하고 활성산소종(ROS)을 생성하는 미토콘드리아 효소입니다.

## cancer_relevance
p53 유도 유전자(PIG6)로, 맥락에 따라 역할이 다릅니다. POX 유래 ROS는 세포자멸사를
유도(종양 억제)할 수 있는 반면, 대사 스트레스 상황에서는 프롤린 이화가 종양세포
생존을 도울 수 있습니다. 효과는 매우 맥락 의존적입니다.

## pathway
프롤린 이화작용, p53 경로, 미토콘드리아 ROS/산화환원 및 대사 재프로그래밍.

## therapeutic_relevance
승인된 표적치료제는 없으며, 대사/산화환원 노드로 연구됩니다. 근거 제한적입니다.

## sources
- GeneCards: https://www.genecards.org/cgi-bin/carddisp.pl?gene=PRODH
- NCBI Gene: https://www.ncbi.nlm.nih.gov/gene/?term=PRODH
- UniProt: https://www.uniprot.org/uniprotkb?query=gene:PRODH+AND+organism_id:9606
- (위 출처는 데이터베이스 집계원이며, 개별 논문 PMID는 심화 큐레이션 시 보강 예정.)

## evidence_limitations
종양 촉진/억제의 양면성이 있어, 본 코호트에서의 암종 특이적 방향성은 근거
제한적입니다.
