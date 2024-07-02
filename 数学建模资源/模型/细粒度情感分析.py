import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = r'C:\Users\USER\OneDrive\桌面\5-1.xlsm'
df = pd.read_excel(file_path)

# 1. 计算整体情感得分
df['Overall_Sentiment'] = df['Review_Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# 2. 定义子情感得分的关键主题
key_topics = ["color", "size","look","price","quality","wear"]
for topic in key_topics:
    df[f'{topic}_Sentiment'] = df['Review_Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity if topic.lower() in str(x).lower() else None)

# 3. 情感分类标签
df['Sentiment_Label'] = pd.cut(df['Overall_Sentiment'], bins=[-1, -0.1, 0.1, 1], labels=['Negative', 'Neutral', 'Positive'])

# 4. 提取关键词分类
def get_keyword_category(text):
    text = str(text).lower()
    if 'quality' in text:
        return 'Product Quality'
    elif 'look' in text:
        return 'Look'
    elif 'color' in text:
        return 'Color'
    elif 'size' in text:
        return 'Size'
    elif 'wear' in text:
        return 'Wear'
    else:
        return 'Other'

df['Keyword_Category'] = df['Review_Text'].apply(get_keyword_category)

# 保存结果到Excel文件
result_file_path = r'C:\Users\USER\OneDrive\桌面\result.xlsm'
df.to_excel(result_file_path, index=False)

