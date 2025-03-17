import numpy as np
import cvxpy as cp
import pandas as pd
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def lmv(X, y, H, alpha):
    num = X.shape[0]  # 样本数量
    r, _ = H.shape  # 投影矩阵的行数
    k = y  # 聚类数

    A = 2 * alpha * np.eye(r) + 2 * H @ H.T
    A = (A + A.T) / 2  # 确保对称
    A = cp.psd_wrap(A)  # 处理可能的非正定问题
    B = X.T  # (d, num)

    S = np.zeros((r, num))

    for ji in range(num):
        ff = -2 * B[:, ji].T @ H.T

        # 定义优化变量
        Z = cp.Variable(r)

        # 构造二次规划问题
        objective = cp.Minimize(0.5 * cp.quad_form(Z, A) + ff @ Z)
        constraints = [cp.sum(Z) == 1, Z >= 0, Z <= 1]
        prob = cp.Problem(objective, constraints)

        # 求解
        prob.solve(solver=cp.SCS, verbose=False)

        S[:, ji] = Z.value

    Sbar = S  # 单视图情况下，无需拼接多个视图

    # 计算 SVD
    U, _, _ = svd(Sbar.T, full_matrices=False)
    U_k = U[:, :k]  # 取前 k 维特征

    # K-means 聚类
    kmeans = KMeans(n_clusters=k, max_iter=100, n_init=10, random_state=5489)
    labels = kmeans.fit_predict(U_k)

    return U_k, labels


def kmeans_sampling(matrix, num_clusters):
    """
    对输入矩阵进行 KMeans 聚类，并在每个簇中选择一个代表性用户。

    参数：
    matrix (np.ndarray): 形状为 (n, m) 的矩阵，其中 n 为用户数，m 为特征维度。
    num_clusters (int): 期望的聚类簇数。

    返回：
    np.ndarray: 选出的代表性用户索引数组。
    """
    matrix = np.asarray(matrix)
    # 运行 KMeans 聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(matrix)
    centroids = kmeans.cluster_centers_

    selected_indices = []

    # 在每个簇中选择离质心最近的样本
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]  # 获取属于该簇的所有索引

        cluster_points = matrix[cluster_indices]  # 取出对应的样本

        # 计算所有样本到质心的欧式距离
        distances = np.linalg.norm(cluster_points - centroids[i], axis=1)

        # 选择距离最近的索引
        closest_index = cluster_indices[np.argmin(distances)]
        selected_indices.append(closest_index)
    return matrix[selected_indices], selected_indices

if __name__ == '__main__':
    # 读取归一化后的负荷数据
    df = pd.read_csv("normalized_load_data.csv")
    # 使用KMeans算法进行聚类，聚类簇数为6
    # 打印出存在空值的列的列号
    nan_columns = df.columns[df.isna().any()].tolist()
    print("Columns with NaN values:", nan_columns)
    df.dropna(axis=1, inplace=True)
    # 转置数据，使得每一行代表一个用户
    df_array_transposed = df.T
    df_array_transposed = np.asarray(df_array_transposed)
    df_array_transposed_part1 = df_array_transposed[:1500]
    df_array_transposed_part2 = df_array_transposed[1500:]

    # cluster_num = [4, 5, 6, 7, 8, 9, 10]
    # alpha = [1, 10, 20, 50, 100]
    # anchor = [10, 20, 30, 40, 80, 100,200]
    cluster_num = [4]
    alpha = [100]
    anchor = [50]
    highest_silhouette_scores = {}
    highest_dbi_scores = {}
    for c in cluster_num:
       highest_silhouette = -1
       best_k_sil = None
       best_a_sil = None
       lowest_dbi = float('inf')
       best_k_dbi = None
       best_a_dbi = None
       for k in anchor:
           for a in alpha:
               print(f"Clusters: {c}, Anchors: {k}, Alpha: {a}")
               sample, anchor_index = kmeans_sampling(df_array_transposed_part1, k)
               U_k, labels = lmv(df_array_transposed_part1, c, sample, a)
               silhouette_avg_pre = silhouette_score(df_array_transposed_part1, labels)
               dbi_avg_pre = davies_bouldin_score(df_array_transposed_part1, labels)
               with open("pre_silhouette_results.txt", "a") as f:
                   f.write(f"Clusters: {c}, Anchors: {k}, Alpha: {a}, Silhouette Index: {silhouette_avg_pre:.4f}, DBI: {dbi_avg_pre:.4f}\n")
               labels_anchor = labels[anchor_index]
               df_array_anchor = sample
               from sklearn.neighbors import KNeighborsClassifier
               knn = KNeighborsClassifier(n_neighbors=3)
               # knn.fit(df_array_anchor, labels_anchor)
               knn.fit(df_array_transposed_part1, labels)
               import time

               print("LMV function execution time")
               start_time = time.time()
               labels_part2 = knn.predict(df_array_transposed_part2)
               end_time = time.time()
               print(f"KNN function execution time: {end_time - start_time} seconds")
               labels = np.concatenate((labels, labels_part2))
               silhouette_avg = silhouette_score(df_array_transposed, labels)
               dbi_avg = davies_bouldin_score(df_array_transposed, labels)
               with open("silhouette_results.txt", "a") as f:
                   f.write(f"Clusters: {c}, Anchors: {k}, Alpha: {a}, Silhouette Index: {silhouette_avg:.4f}, DBI: {dbi_avg:.4f}\n")
               if silhouette_avg > highest_silhouette:
                   highest_silhouette = silhouette_avg
                   best_k_sil = k
                   best_a_sil = a
               if dbi_avg < lowest_dbi:
                   lowest_dbi = dbi_avg
                   best_k_dbi = k
                   best_a_dbi = a
       highest_silhouette_scores[c] = (highest_silhouette, best_k_sil, best_a_sil)
       highest_dbi_scores[c] = (lowest_dbi, best_k_dbi, best_a_dbi)
       with open("highest_silhouette_results.txt", "w") as f:
           for c, (sil_score, k_sil, a_sil) in highest_silhouette_scores.items():
               f.write(f"Clusters: {c}, Best Anchors: {k_sil}, Best Alpha: {a_sil}, Highest Silhouette Index: {sil_score:.4f}\n")

       with open("highest_dbi_results.txt", "w") as f:
           for c, (dbi_score, k_dbi, a_dbi) in highest_dbi_scores.items():
               f.write(f"Clusters: {c}, Best Anchors: {k_dbi}, Best Alpha: {a_dbi}, Lowest DBI: {dbi_score:.4f}\n")


