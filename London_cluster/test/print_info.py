
import pandas as pd
import numpy as np
# Define the file path
file_path = r'G:\论文阅读与写作\负荷聚类letter\论文代码\pythonProject\London_cluster\data\halfhourly_dataset\halfhourly_dataset\block_0.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display basic information about the DataFrame
print(df.info())

# Display the first few rows of the DataFrame
print(df.head())

# Display summary statistics of the DataFrame
print(df.describe())

print("Number of unique LCLid values:", df['LCLid'].nunique())
print("Unique LCLid values:", df['LCLid'].unique())

import os

# Define the directory path
directory_path = r'G:\论文阅读与写作\负荷聚类letter\论文代码\pythonProject\London_cluster\data\halfhourly_dataset\halfhourly_dataset'

# Initialize an empty set to store unique LCLid values
unique_lclids = set()

# Iterate through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        unique_lclids.update(df['LCLid'].unique())

# Print the number of unique LCLid values
print("Total number of unique LCLid values:", len(unique_lclids))
# Print the number of unique LCLid values

np.save('unique_lclids.npy', np.array(list(unique_lclids)))



