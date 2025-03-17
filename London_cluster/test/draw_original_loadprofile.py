import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
pd.read_csv("normalized_load_data.csv")
# 读取归一化后的负荷数据
df = pd.read_csv("normalized_load_data.csv")

# 绘制所有用户的日负荷曲线
# 绘制所有用户的日负荷曲线
for column in df.columns[:50]:
    plt.plot(df[column], linestyle='--', color='blue')

plt.xlabel('Time')
plt.ylabel('Normalized Load')
plt.title('Daily Load Profiles of All Users')
plt.show()