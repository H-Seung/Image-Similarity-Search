import torch


def search_similar(query_vec, db_embeddings, top_k=5):
    """
    cosine similarity 기반 검색 함수 (오류 처리 강화)
    """
    if not db_embeddings:
        print("⚠️ 데이터베이스가 비어있습니다.")
        return []

    if len(db_embeddings) < top_k:
        top_k = len(db_embeddings)
        print(f"⚠️ DB에 {len(db_embeddings)}개 이미지만 있어서 top_k를 {top_k}로 조정합니다.")

    try:
        # 모든 임베딩의 차원을 일관되게 만들기
        db_vecs = []
        for key, vec in db_embeddings.items():
            # 차원 정리: 1D 벡터로 통일
            if vec.dim() > 1:
                vec = vec.squeeze()
            vec = vec.flatten()
            db_vecs.append(vec)

        # 모든 임베딩을 스택으로 결합
        db_tensor = torch.stack(db_vecs)
        file_list = list(db_embeddings.keys())

        # query_vec 차원 정리
        if query_vec.dim() > 1:
            query_vec = query_vec.squeeze()
        query_vec = query_vec.flatten()

        print(f"🔍 검색 정보:")
        print(f"   Query shape: {query_vec.shape}")
        print(f"   DB shape: {db_tensor.shape}")

        # 코사인 유사도 계산
        sims = torch.nn.functional.cosine_similarity(query_vec.unsqueeze(0), db_tensor, dim=1)

        # 상위 k개 결과 선택
        if top_k > 0:
            top_scores, top_indices = torch.topk(sims, k=top_k, largest=True)
            results = [(file_list[idx.item()], score.item()) for idx, score in zip(top_indices, top_scores)]
        else:
            results = []

        return results

    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")
        if 'query_vec' in locals():
            print(f"Query vector shape: {query_vec.shape}")
        if 'db_tensor' in locals():
            print(f"DB tensor shape: {db_tensor.shape}")
        return []