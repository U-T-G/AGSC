import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 读取归一化后的负荷数据
df = pd.read_csv("normalized_load_data.csv")

# 检查并删除存在空值的列
nan_columns = df.columns[df.isna().any()].tolist()
print("Columns with NaN values:", nan_columns)
df.dropna(axis=1, inplace=True)

# 转置数据，使得每一行代表一个用户
df_array_transposed = df.T

# 计算不同聚类簇数的 Silhouette Index
silhouette_scores = {}
dbi_scores = {}
for n_clusters in range(4, 5):
    import time
    print("LMV function execution time")
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, max_iter=100, n_init=10, random_state=5489)
    kmeans.fit(df_array_transposed)
    end_time = time.time()
    print(f"KMeans function execution time: {end_time - start_time} seconds")
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(df_array_transposed, labels)
    silhouette_scores[n_clusters] = silhouette_avg
    dbi_avg = davies_bouldin_score(df_array_transposed, labels)
    dbi_scores[n_clusters] = dbi_avg
    print(f"Clusters: {n_clusters}, Silhouette Index: {silhouette_avg}, DBI: {dbi_avg}")

# 保存结果到txt文件
with open("silhouette_scores_kmeans.txt", "w") as f:
    for n_clusters, score in silhouette_scores.items():
        f.write(f"Clusters: {n_clusters}, Silhouette Index: {silhouette_scores[n_clusters]}, DBI: {dbi_scores[n_clusters]}\n")

print("Silhouette scores saved to silhouette_scores.txt")
