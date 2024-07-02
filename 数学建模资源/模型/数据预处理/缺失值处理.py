import pandas as pd

# 设置文件路径
file_path = r'C:\Users\USER\OneDrive\桌面\raw_data.csv'

# 加载数据
try:
    data = pd.read_csv(file_path)
    print("数据已成功加载！")
except Exception as e:
    print("加载数据时出错：", e)

# 定义同时为 "null" 的缺失值条件
missing_condition = (data['Review_Text'] == "null") & (data['Title'] == "null")

# 计算同时为 "null" 的行数
missing_rows_count = missing_condition.sum()
print(f"将要删除 {missing_rows_count} 行，因为 Review_Text 和 Title 同时为 'null'。")

# 删除同时为 "null" 的行
data = data[~missing_condition]

# 输出删除后的数据信息
print(f"删除后的数据还剩 {len(data)} 行。")