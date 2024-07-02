import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# 读取Excel数据
file_path = r'C:\Users\USER\OneDrive\桌面\5-1.xlsm'
df = pd.read_excel(file_path)

# 获取评论文本数据
reviews = df['Review_Text'].tolist()

# 数据预处理，包括去停用词、词形还原等
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]  # 去除非字母字符
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # 词形还原和去停用词
    return words

processed_reviews = [preprocess_text(review) for review in reviews]

# 创建文档-词矩阵
dictionary = corpora.Dictionary(processed_reviews)
corpus = [dictionary.doc2bow(review) for review in processed_reviews]

# 使用gensim进行LDA主题建模
num_topics = 5  # 指定主题数
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# 输出主题词
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)
