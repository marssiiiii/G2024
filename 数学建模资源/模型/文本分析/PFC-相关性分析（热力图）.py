1.import pandas as pd  
2.from collections import Counter  
3.from nltk.corpus import stopwords  
4.from nltk.tokenize import word_tokenize  
5.  
6.# 下载NLTK的停用词列表  
7.import nltk  
8.nltk.download('punkt')  
9.  
10.# 读取Excel文件  
11.file_path = r'C:\Users\USER\OneDrive\桌面\5-1.xlsm'  
12.df = pd.read_excel(file_path)  
13.  
14.# 提取Positive Feedback Count和Review_Text列  
15.feedback_count = df['Positive_Feedback_Count']  
16.reviews = df['Review_Text']  
17.  
18.# 将评论文本合并为一个字符串  
19.all_text = ' '.join(reviews.astype(str))  
20.  
21.# 分词  
22.words = word_tokenize(all_text)  
23.  
24.# 移除停用词  
25.stop_words = set(stopwords.words('english'))  
26.filtered_words = [word for word in words if word.lower() not in stop_words]  
27.  
28.# 计算词频  
29.word_count = Counter(filtered_words)  
30.  
31.# 提取出现频率较高的关键词，你可以根据需要调整top_n的值  
32.top_n = 10  
33.most_common_words = word_count.most_common(top_n)  
34.  
35.# 打印结果  
36.for word, count in most_common_words:  
37.    print(f'{word}: {count}')  