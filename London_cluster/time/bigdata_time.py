import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
import time
import LMVSC
data = np.load("GAN_data.npy")

# KMeans聚类
start_time = time.time()
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(data)
end_time = time.time()
print(f"KMeans clustering execution time: {end_time - start_time} seconds")

# AGSC
sample = LMVSC.kmeans_sampling(data, 20)
print(sample.shape)
import time

print("LMV function execution time")
start_time = time.time()
U_k, labels = LMVSC.lmv(data, 5, sample, 100)
end_time = time.time()
print(f"LMV function execution time: {end_time - start_time} seconds")
# 层次聚类
start_time = time.time()
agglomerative = AgglomerativeClustering(n_clusters=5)
agglomerative.fit(data)
end_time = time.time()
print(f"Agglomerative Clustering execution time: {end_time - start_time} seconds")