import torch

def search_similar(query_vec, db_embeddings, top_k=5):
    """
     cosine similarity 기반 검색 함수
    """
    db_tensor = torch.stack(list(db_embeddings.values()))
    file_list = list(db_embeddings.keys())

    sims = torch.nn.functional.cosine_similarity(query_vec.unsqueeze(0), db_tensor)
    top_scores, top_indices = torch.topk(sims, k=top_k)

    results = [(file_list[i], top_scores[idx].item()) for idx, i in enumerate(top_indices)]
    return results