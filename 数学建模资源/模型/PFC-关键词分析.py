import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string

# 读取Excel文件
file_path = r'C:\Users\USER\OneDrive\桌面\data.xlsm'
df = pd.read_excel(file_path)

# 合并评论中的文本
merged_text = ' '.join(df['Review_Text'].astype(str))

# 分词并去除停用词
stop_words = set(stopwords.words('english'))
tokens = word_tokenize(merged_text.lower())
filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words and word not in string.punctuation]

# 计算单词频率
fdist = FreqDist(filtered_tokens)

# 输出前N个单词
top_words = 10
print(f"Top {top_words} words in Review_Text:")
print(fdist.most_common(top_words))
