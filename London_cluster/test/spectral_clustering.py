import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 读取归一化后的负荷数据
df = pd.read_csv("normalized_load_data.csv")

# 检查并删除存在空值的列
nan_columns = df.columns[df.isna().any()].tolist()
print("Columns with NaN values:", nan_columns)
df.dropna(axis=1, inplace=True)

# 转置数据，使得每一行代表一个用户
df_array_transposed = df.T

# 计算不同聚类簇数的 Silhouette Index 和 DBI
silhouette_scores = {}
dbi_scores = {}
for n_clusters in range(4, 11):
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')
    labels = spectral.fit_predict(df_array_transposed)
    silhouette_avg = silhouette_score(df_array_transposed, labels)
    dbi_avg = davies_bouldin_score(df_array_transposed, labels)
    silhouette_scores[n_clusters] = silhouette_avg
    dbi_scores[n_clusters] = dbi_avg
    print(f"Clusters: {n_clusters}, Silhouette Index: {silhouette_avg}, DBI: {dbi_avg}")

# 保存结果到txt文件
with open("clustering_scores_spectral.txt", "w") as f:
    for n_clusters in silhouette_scores.keys():
        f.write(f"Clusters: {n_clusters}, Silhouette Index: {silhouette_scores[n_clusters]}, DBI: {dbi_scores[n_clusters]}\n")

print("Clustering scores saved to clustering_scores_spectral.txt")
