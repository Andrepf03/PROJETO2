import numpy as np
from sklearn.neighbors import NearestNeighbors


class SearchSpace:
    def __init__(self, name, transform, X_gallery, metric="euclidean"):
        self.name = name
        self.transform = transform
        self.gallery_features = transform(X_gallery)                
        self.nn = NearestNeighbors(metric=metric, algorithm="brute").fit(self.gallery_features)

    def search(self, X_query, k=10):
        X_query = np.atleast_2d(X_query)                             
        Q = self.transform(X_query)                                   
        distances, indices = self.nn.kneighbors(Q, n_neighbors=k)
        return indices, distances


def retrieve(X_query, spaces, k=10):
    return {space.name: space.search(X_query, k) for space in spaces}




def precision_at_k(retrieved_idx, gallery_labels, query_labels, k):
    top_k_labels = gallery_labels[retrieved_idx[:, :k]]           
    hits = top_k_labels == query_labels[:, None]                
    return hits.mean(axis=1).mean()                              