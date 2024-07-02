import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# 下载VADER情感分析器所需资源
nltk.download('vader_lexicon')

# 读取Excel文件
file_path = r'C:\Users\USER\OneDrive\桌面\5-1.xlsm'
df = pd.read_excel(file_path)

# 初始化Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# 执行情感分析并将得分添加到'K'列
df['Sentiment_Score'] = df['Review_Text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# 手动调整分类规则
# 这里的规则是根据得分范围来划分，你可以根据需要调整这些范围
df['Sentiment_Category'] = df['Sentiment_Score'].apply(
    lambda x: 'positive' if x >= 0.78 else ('neutral' if 0.48 <= x < 0.78 else 'negative')
)

# 将更新后的DataFrame保存到Excel文件
df.to_excel(file_path, index=False)
