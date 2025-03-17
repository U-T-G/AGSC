import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 读取归一化后的负荷数据
df = pd.read_csv("normalized_load_data.csv")
# 使用KMeans算法进行聚类，聚类簇数为6
# 打印出存在空值的列的列号
nan_columns = df.columns[df.isna().any()].tolist()
print("Columns with NaN values:", nan_columns)
df.dropna(axis=1, inplace=True)
# 转置数据，使得每一行代表一个用户
df_array_transposed = df.T

from sklearn.cluster import AgglomerativeClustering# 进行层次聚类
import time
print("LMV function execution time")
start_time = time.time()
clustering = AgglomerativeClustering(n_clusters=4)
labels = clustering.fit_predict(df_array_transposed)
end_time = time.time()
print(f"Agglomerative Clustering function execution time: {end_time - start_time} seconds")
from sklearn.metrics import silhouette_score

# 计算聚类结果的Silhouette Index
silhouette_avg = silhouette_score(df_array_transposed, labels)
print(f"Silhouette Index: {silhouette_avg}")