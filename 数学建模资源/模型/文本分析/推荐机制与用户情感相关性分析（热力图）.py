import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 读取Excel文件
file_path = r'C:\Users\USER\OneDrive\桌面\5-1.xlsm'
df = pd.read_excel(file_path)

# 将Sentiment_Category列进行独热编码
df_encoded = pd.get_dummies(df['Sentiment_Category'], prefix='Sentiment_Category')

# 合并编码后的列到原始DataFrame
df = pd.concat([df, df_encoded], axis=1)

# 计算皮尔逊相关系数
correlation_sentiment_score = pearsonr(df['Sentiment_Score'], df['Recommended_IND'])
correlation_sentiment_positive = pearsonr(df['Sentiment_Category_positive'], df['Recommended_IND'])
correlation_sentiment_neutral = pearsonr(df['Sentiment_Category_neutral'], df['Recommended_IND'])
correlation_sentiment_negative = pearsonr(df['Sentiment_Category_negative'], df['Recommended_IND'])

# 输出皮尔逊相关系数
print(f"Pearson correlation between Sentiment_Score and Recommended_IND: {correlation_sentiment_score[0]:.2f}")
print(f"Pearson correlation between Sentiment_Category_positive and Recommended_IND: {correlation_sentiment_positive[0]:.2f}")
print(f"Pearson correlation between Sentiment_Category_neutral and Recommended_IND: {correlation_sentiment_neutral[0]:.2f}")
print(f"Pearson correlation between Sentiment_Category_negative and Recommended_IND: {correlation_sentiment_negative[0]:.2f}")

# 绘制热力图
correlation_matrix = df[['Sentiment_Score', 'Recommended_IND', 'Sentiment_Category_positive', 'Sentiment_Category_neutral', 'Sentiment_Category_negative']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
