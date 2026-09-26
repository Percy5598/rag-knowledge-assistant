import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def retrieve(
    query,
    documents,
    document_embeddings,
    model,
    top_k = 3,
): 
    """
    Retrieve the most relevant documents for a query.
    """ 
    query_embedding = model.encode([query])
    scores = cosine_similarity(
        query_embedding,
        document_embeddings,
    )[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for index in top_indices:
        results.append(
            {
                "text": documents[index],
                "score": float(scores[index]), 
            }
        )


    return results 