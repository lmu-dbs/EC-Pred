from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

class ECFactory:
    _clustering = {
        "GaussianMixture": GaussianMixture,
        "KMeans": KMeans,
    }

    @classmethod
    def create(cls, clustering, **cluster_params):
        if clustering in cls._clustering:
            clustering_class = cls._clustering[clustering]
            return clustering_class(**cluster_params)
        else:
            raise ValueError(f"Unknown clustering:{clustering}")