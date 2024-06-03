import keras
import numpy as np
import tensorflow as tf
import keras.layers as layers
from keras import optimizers,losses

batches = 128
total_words = 10000
max_review_len = 80  # 每个句子的大长度
embedding_len = 100  # 词向量的最大长度

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
#print(x_train.shape, len(x_train[0]))  # x为list构成的向量，每个list代表一个句子不定长度
#print(y_train.shape, y_train)

# 填充句子
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

# 构建数据集（打散，分批）
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batches, drop_remainder=True)  # batches为每个批次的样本数
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batches, drop_remainder=True)


# 定义网络模型
class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN,self).__init__()
        self.state0=tf.zeros([batches,units])
        self.state1=tf.zeros([batches,units])
        self.units=units
        self.embedding = (layers.
                          Embedding(total_words, embedding_len, input_length=max_review_len))
        # 构建cell
        self.rnn_cell0 = layers.SimpleRNNCell(units)
        self.rnn_cell1 = layers.SimpleRNNCell(units)

        self.out_layer = layers.Dense(1)

    def call(self, inputs, training=None):
            global out1
            x = self.embedding(inputs) #将隐藏的词向量转换出来，[b,80]->[b,80,100]
            state0=self.state0
            state1=self.state1
            for words in tf.unstack(x, axis=1):
                out0, [state0] = self.rnn_cell0(words, [state0], training) #注意每次要进行打包和解包
                out1, [state1] = self.rnn_cell1(state0, [state1], training)
            x = self.out_layer(out1)
            prob = tf.sigmoid(x)
            return prob


def main():
    units = 64
    epochs = 10
    model = MyRNN(units)
    model.compile(optimizer=optimizers.Adam(0.01),loss=losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    history=model.fit(db_train,epochs=epochs,validation_data=db_test) #fit函数会在内部调用call函数
    model.evaluate(db_test)

if __name__=="__main__":
    main()