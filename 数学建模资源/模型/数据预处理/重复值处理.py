import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 设置文件路径
file_path = r'C:\Users\USER\OneDrive\桌面\2.csv'

# 加载数据
try:
    data = pd.read_csv(file_path)
    print("数据已成功加载！")
except Exception as e:
    print("加载数据时出错：", e)

# 首先，确保 Review_Text 列的数据类型是字符串
data['Review_Text'] = data['Review_Text'].astype(str)

# 使用 TF-IDF 计算文本特征
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['Review_Text'])

# 计算余弦相似度
cosine_sim = cosine_similarity(tfidf_matrix)

# 查找相似度高于95%的行
similarity_threshold = 0.95
# 创建一个用于标记删除的数组
to_remove = np.zeros(len(data), dtype=bool)

for i in range(len(data)):
    if not to_remove[i]:  # 如果这一行还未被标记为删除
        # 获取与当前行相似度超过95%的其他行的索引
        similar_indices = np.where((cosine_sim[i] > similarity_threshold) & (data['Clothing_ID'] == data['Clothing_ID'][i]))[0]
        # 标记这些行为删除，除了i本身
        to_remove[similar_indices] = True
        to_remove[i] = False  # 保留当前行

# 删除标记的行
data = data[~to_remove]
print(f"删除后的数据还剩 {len(data)} 行。")

# 将处理后的数据保存回原文件
try:
    data.to_csv(file_path, index=False)
    print(f"处理后的数据已保存回 {file_path}")
except Exception as e:
    print(f"保存文件时出错：{e}")
