import pandas as pd

# 设置文件路径
file_path = r'C:\Users\USER\OneDrive\桌面\2.csv'

# 加载数据
try:
    data = pd.read_csv(file_path)
    print("数据已成功加载！")
except Exception as e:
    print("加载数据时出错：", e)

# 处理 Age 列的异常值，设定有效年龄范围为 18 到 80
original_age_count = data.shape[0]
data = data[(data['Age'] >= 18) & (data['Age'] <= 80)]
filtered_age_count = data.shape[0]
print(f"移除年龄不在18至80岁之间的数据，共删除 {original_age_count - filtered_age_count} 行。")

# 对 Positive_Feedback_Count 进行异常值处理
Q1 = data['Positive_Feedback_Count'].quantile(0.25)
Q3 = data['Positive_Feedback_Count'].quantile(0.75)
IQR = Q3 - Q1

# 定义异常值的上下界
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 显示原始数据中异常值的数量
print(f"原始数据中点赞数异常值的数量：{data[(data['Positive_Feedback_Count'] < lower_bound) | (data['Positive_Feedback_Count'] > upper_bound)].shape[0]}")

# 过滤掉异常值
data = data[(data['Positive_Feedback_Count'] >= lower_bound) & (data['Positive_Feedback_Count'] <= upper_bound)]

# 显示处理后的数据信息
print(f"处理后的数据还剩 {len(data)} 行。")

# 将处理后的数据保存回原文件
try:
    data.to_csv(file_path, index=False)
    print(f"处理后的数据已保存回 {file_path}")
except Exception as e:
    print(f"保存文件时出错：{e}")
