import pandas as pd
import glob
import os

# 获取所有 CSV 文件路径
all_files = glob.glob(
    r"G:\论文阅读与写作\负荷聚类letter\论文代码\pythonProject\London_cluster\data\halfhourly_dataset\halfhourly_dataset\*.csv"
)

# 结果文件路径
output_file = r"G:\论文阅读与写作\负荷聚类letter\论文代码\pythonProject\London_cluster\test\daily_avg_load_curve.csv"

# 初始化结果 DataFrame
final_df = None

# 遍历每个 CSV 文件
for file in all_files:
    print(f"正在处理文件: {file}")  # 显示当前处理的文件
    # 读取 CSV 文件
    df = pd.read_csv(file, parse_dates=["tstp"])

    # 规范列名
    df.columns = ["LCLid", "tstp", "energy(kWh/hh)"]
    df["energy(kWh/hh)"] = pd.to_numeric(df["energy(kWh/hh)"], errors="coerce")  # 转换为数值类型

    # 提取时间信息
    df["date"] = df["tstp"].dt.date  # 仅保留日期
    df["time"] = df["tstp"].dt.time  # 仅保留时间（00:00, 00:30, ..., 23:30）

    # 计算该文件的日均负荷曲线（按用户和时间取平均值）
    daily_avg_curve = (
        df.groupby(["LCLid", "time"])["energy(kWh/hh)"]
        .mean()
        .reset_index()
        .pivot(index="time", columns="LCLid", values="energy(kWh/hh)")
    )

    # 处理 NaN 值（部分时间点可能缺失）
    daily_avg_curve = daily_avg_curve.dropna(how="all")  # 仅在所有用户均为空时删除该行

    # **合并数据**（按时间索引对齐，新增用户列）
    if final_df is None:
        final_df = daily_avg_curve  # 第一个文件，直接赋值
    else:
        final_df = final_df.join(daily_avg_curve, how="outer")  # 以时间索引对齐，添加新用户列

    print(f"文件 {file} 处理完成")

# **最终保存结果**
if final_df is not None:
    final_df.to_csv(output_file)
    print(f"所有文件处理完成，已保存到 {output_file}")
else:
    print("未找到有效数据，未生成 CSV 文件")
