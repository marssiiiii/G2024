import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 下载NLTK的停用词列表
import nltk
nltk.download('punkt')

# 读取Excel文件
file_path = r'C:\Users\USER\OneDrive\桌面\5-1.xlsm'
df = pd.read_excel(file_path)

# 提取Positive Feedback Count和Review_Text列
feedback_count = df['Positive_Feedback_Count']
reviews = df['Review_Text']

# 将评论文本合并为一个字符串
all_text = ' '.join(reviews.astype(str))

# 分词
words = word_tokenize(all_text)

# 移除停用词
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]

# 计算词频
word_count = Counter(filtered_words)

# 提取出现频率较高的关键词，你可以根据需要调整top_n的值
top_n = 10
most_common_words = word_count.most_common(top_n)

# 打印结果
for word, count in most_common_words:
    print(f'{word}: {count}')
