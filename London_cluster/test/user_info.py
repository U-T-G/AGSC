import pandas as pd
import glob
import os

pd.read_csv("daily_avg_load_curve.csv")

# 读取 CSV 文件
df = pd.read_csv("daily_avg_load_curve.csv", index_col=0)

# 获取负荷数据（去除时间戳列）
load_data = df.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce')

for col in load_data.columns:
    for i in range(len(load_data)):
        if load_data[col].iloc[i] == 0:
            if i == 0:  # 如果是第一行，替换为下一行的值
                print(i)
                load_data[col].iloc[i] = load_data[col].iloc[i + 1]
            else:  # 否则替换为上一行的值
                load_data[col].iloc[i] = load_data[col].iloc[i - 1]

# 使用min-max归一化
normalized_data = (load_data - load_data.min().min()) / (load_data.max().max() - load_data.min().min())


# 保存为n*m的数组
n_m_array = normalized_data.values

# 保存为csv文件
normalized_data.to_csv("normalized_load_data.csv", index=False)
